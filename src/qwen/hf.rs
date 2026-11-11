use half::{bf16, f16};
use hf_hub::api::sync::Api;
use memmap2::MmapOptions;
use safetensors::{tensor::TensorView, Dtype, SafeTensors, View};
use serde::Deserialize;
use std::{
    borrow::Cow,
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

/// File name of the prepared (combined, bf16) weights inside the HF snapshot directory.
pub const COMBINED_WEIGHTS_FILE: &str = "model_combined_bf16_v1.safetensors";

/// Index file structure for sharded safetensors models
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes} bytes")
    }
}

/// Convert tensor data to an f32 vec.
fn tensor_to_f32(tensor: &safetensors::tensor::TensorView) -> Vec<f32> {
    let dtype = tensor.dtype();
    let data = tensor.data();

    match dtype {
        Dtype::F32 => bytemuck::cast_slice::<u8, f32>(data).to_vec(),
        Dtype::F16 => {
            let f16_slice: &[f16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let bf16_slice: &[bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        other => {
            panic!("Unsupported dtype for conversion: {other:?}");
        }
    }
}

fn tensor_to_f32_bytes(tensor: &safetensors::tensor::TensorView) -> Vec<u8> {
    let fp32 = tensor_to_f32(tensor);
    bytemuck::cast_slice(&fp32).to_vec()
}

fn tensor_to_bf16_bytes(tensor: &safetensors::tensor::TensorView) -> Vec<u8> {
    match tensor.dtype() {
        Dtype::BF16 => tensor.data().to_vec(),
        _ => tensor_to_f32(tensor)
            .into_iter()
            .flat_map(|x| bf16::from_f32(x).to_le_bytes())
            .collect(),
    }
}

/// Norm weights stay F32 in the bf16 pipeline: the model computes norms
/// (including the per-head QK-norm) in F32 via explicit casts, while the
/// linear/embedding weights are bf16. `q_norm`/`k_norm` contain "norm", so the
/// substring check keeps them F32 too.
fn keep_f32_in_bf16_pipeline(name: &str) -> bool {
    name.contains("norm")
}

/// A source tensor re-encoded for the combined file — lazily: `data()` converts on demand, so
/// the writer streams one tensor at a time straight from the mmapped source shards and peak RAM
/// is a single tensor, not the whole model (the old path held every tensor *and* the serialized
/// file in memory — ~2x model size, enough to OOM a default-sized WSL2 VM on Qwen3-4B).
struct CombinedTensor<'a> {
    src: TensorView<'a>,
    dtype: Dtype,
}

impl<'a> CombinedTensor<'a> {
    fn new(name: &str, src: TensorView<'a>) -> Self {
        let dtype = if keep_f32_in_bf16_pipeline(name) { Dtype::F32 } else { Dtype::BF16 };
        Self { src, dtype }
    }
}

impl View for CombinedTensor<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        self.src.shape()
    }
    fn data(&self) -> Cow<'_, [u8]> {
        if self.src.dtype() == self.dtype {
            Cow::Borrowed(self.src.data())
        } else if self.dtype == Dtype::F32 {
            Cow::Owned(tensor_to_f32_bytes(&self.src))
        } else {
            Cow::Owned(tensor_to_bf16_bytes(&self.src))
        }
    }
    fn data_len(&self) -> usize {
        let elems: usize = self.src.shape().iter().product();
        elems * if self.dtype == Dtype::F32 { 4 } else { 2 }
    }
}

/// Whether `path` is a complete safetensors file: the header parses and the file is exactly as
/// long as the header says. Catches a combine that was interrupted mid-write.
fn is_complete_safetensors(path: &Path) -> bool {
    let Ok(file) = File::open(path) else { return false };
    let Ok(mmap) = (unsafe { MmapOptions::new().map(&file) }) else { return false };
    // `deserialize` validates the header and that the tensor offsets exactly cover the buffer.
    SafeTensors::deserialize(&mmap).is_ok()
}

fn model_shard_files(model_dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_shard_path = model_dir.join("model.safetensors");

    if single_shard_path.exists() && !index_path.exists() {
        Ok(vec![single_shard_path])
    } else if index_path.exists() {
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

        let mut files: Vec<String> = index.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();

        Ok(files.into_iter().map(|f| model_dir.join(f)).collect())
    } else {
        Err("No model.safetensors or model.safetensors.index.json found".into())
    }
}

/// Combines sharded safetensors into a single bf16 file (norm weights kept
/// F32). Linear projections, the token embedding and the lm_head are bf16; the
/// model is pure HLIR and the e-graph rewrites pick up the bf16 matmuls.
///
/// Qwen3-4B ties the lm_head to the embedding table, so no separate
/// `lm_head.weight` is stored — the model reuses `model.embed_tokens.weight`.
pub fn combine_safetensors_to_bf16(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join(COMBINED_WEIGHTS_FILE);

    if output_path.exists() {
        if is_complete_safetensors(&output_path) {
            let existing_bytes = std::fs::metadata(&output_path)?.len();
            tracing::info!(
                "using existing combined weights at {} ({})",
                output_path.display(),
                format_bytes(existing_bytes)
            );
            return Ok(output_path);
        }
        tracing::warn!(
            "combined weights at {} are incomplete or corrupt (interrupted run?) — rebuilding",
            output_path.display()
        );
        std::fs::remove_file(&output_path)?;
    }

    let shard_files = model_shard_files(model_dir)?;
    tracing::info!("combining {} weight file(s) into bf16 (norms f32)…", shard_files.len());

    // Keep every source shard mmapped for the duration; the tensors below borrow from them.
    let mut mmaps = Vec::with_capacity(shard_files.len());
    for shard_path in &shard_files {
        let file = File::open(shard_path)?;
        mmaps.push(unsafe { MmapOptions::new().map(&file)? });
    }
    let sources = mmaps
        .iter()
        .map(|m| SafeTensors::deserialize(m))
        .collect::<Result<Vec<_>, _>>()?;

    // Later shards win on a duplicate name (matches the old HashMap-insert behaviour).
    let mut tensors: HashMap<String, CombinedTensor<'_>> = HashMap::new();
    for st in &sources {
        for name in st.names() {
            tensors.insert(name.to_string(), CombinedTensor::new(name, st.tensor(name)?));
        }
    }
    let total: usize = tensors.values().map(|t| t.data_len()).sum();
    tracing::info!(
        "writing {} tensors ({}) to {}…",
        tensors.len(),
        format_bytes(total as u64),
        output_path.display()
    );

    // Write to a temp name and rename: a crash or Ctrl-C mid-write must never leave a truncated
    // file under the final name (it would be picked up as "existing" on the next run).
    let tmp_path = model_dir.join(format!("{COMBINED_WEIGHTS_FILE}.{}.tmp", std::process::id()));
    let written = safetensors::serialize_to_file(tensors, None, &tmp_path)
        .map_err(Box::<dyn std::error::Error>::from)
        .and_then(|()| Ok(File::open(&tmp_path)?.sync_all()?))
        .and_then(|()| Ok(std::fs::rename(&tmp_path, &output_path)?));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    tracing::info!("combined weights saved");
    Ok(output_path)
}

/// Downloads model files from HuggingFace and returns the cache directory path.
pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();
    // Sharded repos have an index; single-file repos don't. Only a failed *index* fetch means
    // "single file" — a failed weight download must surface (it used to be read as "sharded",
    // and the real network error was replaced by a confusing missing-index one).
    let index_path = match repo.get("model.safetensors.index.json") {
        Ok(p) => p,
        Err(_) => {
            repo.get("model.safetensors")?;
            return Ok(model_dir);
        }
    };
    // Parse index to find shard files
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;
    // Get unique shard files
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();
    // Download each shard
    for shard_file in &shard_files {
        repo.get(shard_file)?;
    }
    Ok(model_dir)
}

/// Downloads a model from HuggingFace and prepares it for use.
///
/// Returns the path to the model directory containing:
/// - tokenizer.json
/// - [`COMBINED_WEIGHTS_FILE`]
pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors_to_bf16(&model_dir)?;
    Ok(model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Write a 2-shard fake model (f32 + bf16 sources, one norm weight) into a temp dir.
    fn fake_model(dir: &Path) {
        let w0 = f32_bytes(&[1.0, -2.5, 0.125, 3.0]);
        let norm = f32_bytes(&[0.5, 1.5]);
        let w1: Vec<u8> = [4.0f32, -8.0].iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect();
        let a = [
            ("model.layers.0.mlp.up_proj.weight", TensorView::new(Dtype::F32, vec![2, 2], &w0).unwrap()),
            ("model.layers.0.input_layernorm.weight", TensorView::new(Dtype::F32, vec![2], &norm).unwrap()),
        ];
        let b = [("model.layers.1.mlp.up_proj.weight", TensorView::new(Dtype::BF16, vec![2], &w1).unwrap())];
        safetensors::serialize_to_file(a, None, &dir.join("model-00001-of-00002.safetensors")).unwrap();
        safetensors::serialize_to_file(b, None, &dir.join("model-00002-of-00002.safetensors")).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {
                "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.mlp.up_proj.weight": "model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("potpori-hf-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn combine_converts_and_is_atomic() {
        let dir = temp_dir("combine");
        fake_model(&dir);
        let out = combine_safetensors_to_bf16(&dir).unwrap();
        assert_eq!(out, dir.join(COMBINED_WEIGHTS_FILE));

        let bytes = std::fs::read(&out).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(st.names().len(), 3);
        // f32 linear -> bf16 (these values are exactly representable)
        let up0 = st.tensor("model.layers.0.mlp.up_proj.weight").unwrap();
        assert_eq!((up0.dtype(), up0.shape()), (Dtype::BF16, &[2usize, 2][..]));
        assert_eq!(tensor_to_f32(&up0), vec![1.0, -2.5, 0.125, 3.0]);
        // norm stays f32
        let norm = st.tensor("model.layers.0.input_layernorm.weight").unwrap();
        assert_eq!(norm.dtype(), Dtype::F32);
        assert_eq!(tensor_to_f32(&norm), vec![0.5, 1.5]);
        // bf16 passes through untouched
        let up1 = st.tensor("model.layers.1.mlp.up_proj.weight").unwrap();
        assert_eq!(up1.dtype(), Dtype::BF16);
        assert_eq!(tensor_to_f32(&up1), vec![4.0, -8.0]);

        // No temp file left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn truncated_combined_file_is_rebuilt() {
        let dir = temp_dir("truncated");
        fake_model(&dir);
        let out = combine_safetensors_to_bf16(&dir).unwrap();
        let good = std::fs::read(&out).unwrap();

        // Simulate an interrupted write under the final name (what the old code could leave).
        std::fs::write(&out, &good[..good.len() - 3]).unwrap();
        assert!(!is_complete_safetensors(&out));
        let out2 = combine_safetensors_to_bf16(&dir).unwrap();
        assert_eq!(std::fs::read(&out2).unwrap(), good, "must rebuild to the same bytes");

        // A complete file is reused as-is (not rewritten).
        let before = std::fs::metadata(&out).unwrap().modified().unwrap();
        combine_safetensors_to_bf16(&dir).unwrap();
        assert_eq!(std::fs::metadata(&out).unwrap().modified().unwrap(), before);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

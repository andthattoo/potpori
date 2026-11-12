//! Weight download and preparation.
//!
//! A model's Hugging Face safetensors files are combined once into a single prepared file
//! ([`COMBINED_WEIGHTS_FILE`], next to the download in the HF cache): matrices as BF16, norm
//! weights as F32. Each shard then loads only its own parameters from that file.

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

/// File name of the prepared weights inside the model's HF snapshot directory.
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
        Dtype::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        Dtype::BF16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
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

/// A source tensor re-encoded for the prepared file, lazily: `data()` converts on demand, so
/// the writer streams one tensor at a time from the memory-mapped downloads and peak memory
/// is a single tensor rather than the whole model.
struct CombinedTensor<'a> {
    src: TensorView<'a>,
    dtype: Dtype,
}

impl<'a> CombinedTensor<'a> {
    fn new(name: &str, src: TensorView<'a>) -> Self {
        let dtype = if keep_f32_in_bf16_pipeline(name) {
            Dtype::F32
        } else {
            Dtype::BF16
        };
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
        let elements: usize = self.src.shape().iter().product();
        elements * if self.dtype == Dtype::F32 { 4 } else { 2 }
    }
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

/// Combine the downloaded safetensors files into the single prepared file (matrices BF16,
/// norm weights F32; the graph converts to F32 arithmetic explicitly). A complete prepared
/// file from an earlier run is reused.
///
/// Qwen3 ties the output projection to the embedding table, so no separate `lm_head.weight`
/// is stored — the model reuses `model.embed_tokens.weight`.
pub fn combine_safetensors_to_bf16(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join(COMBINED_WEIGHTS_FILE);

    if valid_combined_file(&output_path) {
        tracing::info!(
            "using prepared weights at {} ({})",
            output_path.display(),
            format_bytes(std::fs::metadata(&output_path)?.len())
        );
        return Ok(output_path);
    }

    // Keep every downloaded file mapped while writing; the tensors below borrow from them.
    let shard_files = model_shard_files(model_dir)?;
    let mut mmaps = Vec::with_capacity(shard_files.len());
    for shard_path in &shard_files {
        let file = File::open(shard_path)?;
        // SAFETY: files in the HF cache are not modified while this process reads them.
        mmaps.push(unsafe { MmapOptions::new().map(&file)? });
    }
    let sources = mmaps
        .iter()
        .map(|m| SafeTensors::deserialize(m))
        .collect::<Result<Vec<_>, _>>()?;
    let mut tensors: HashMap<String, CombinedTensor<'_>> = HashMap::new();
    for st in &sources {
        for name in st.names() {
            tensors.insert(
                name.to_string(),
                CombinedTensor::new(name, st.tensor(name)?),
            );
        }
    }
    let total: usize = tensors.values().map(|t| t.data_len()).sum();
    tracing::info!(
        "preparing weights: {} tensors from {} file(s), {} → {}",
        tensors.len(),
        shard_files.len(),
        format_bytes(total as u64),
        output_path.display()
    );

    // Write under a temporary name and rename: an interrupted run must never leave a
    // truncated file under the final name.
    let temp = tempfile::NamedTempFile::new_in(model_dir)?;
    safetensors::serialize_to_file(tensors, None, temp.path())?;
    File::open(temp.path())?.sync_all()?;
    temp.persist(&output_path).map_err(|e| e.error)?;
    Ok(output_path)
}

/// Whether `path` is a complete safetensors file: the header parses and the tensor offsets
/// cover the file exactly.
fn valid_combined_file(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    // SAFETY: files published by this module are immutable after an atomic rename.
    let Ok(mmap) = (unsafe { MmapOptions::new().map(&file) }) else {
        return false;
    };
    SafeTensors::deserialize(&mmap).is_ok_and(|st| !st.is_empty())
}

/// Load only this shard's declared parameter ports. Byte decoding does not assume mmap alignment.
pub fn load_inputs(
    graph: &super::model::QwenGraph,
    path: &Path,
) -> anyhow::Result<super::model::Inputs> {
    use super::model::{Inputs, TensorData};
    use luminal::dtype::DType;
    let file = File::open(path)?;
    // SAFETY: the prepared model is immutable and remains mapped throughout decoding.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    let mut inputs = Inputs::default();
    for p in &graph.parameters {
        let tensor = tensors.tensor(&p.name)?;
        anyhow::ensure!(tensor.shape() == p.shape, "shape mismatch for {}", p.name);
        let bits = |tensor: &safetensors::tensor::TensorView| -> Vec<u16> {
            tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        // Linear layers are held `[in, out]` in the graph; the checkpoint has `[out, in]`.
        let (rows, cols) = match (p.transpose, &p.shape[..]) {
            (true, &[rows, cols]) => (rows, cols),
            (true, _) => anyhow::bail!("{} is not a matrix", p.name),
            (false, _) => (0, 0),
        };
        let data = match p.dtype {
            // A checkpoint used as published may store a matrix in another float type.
            DType::Bf16 if tensor.dtype() == Dtype::BF16 => {
                TensorData::BF16(turned(bits(&tensor), rows, cols))
            }
            DType::Bf16 => TensorData::BF16(turned(
                tensor_to_f32(&tensor)
                    .into_iter()
                    .map(|x| bf16::from_f32(x).to_bits())
                    .collect(),
                rows,
                cols,
            )),
            DType::F16 if tensor.dtype() == Dtype::F16 => {
                TensorData::F16(turned(bits(&tensor), rows, cols))
            }
            DType::F16 => TensorData::F16(turned(
                tensor_to_f32(&tensor)
                    .into_iter()
                    .map(|x| f16::from_f32(x).to_bits())
                    .collect(),
                rows,
                cols,
            )),
            DType::F32 => TensorData::F32(turned(tensor_to_f32(&tensor), rows, cols)),
            _ => anyhow::bail!("unsupported parameter dtype"),
        };
        inputs.insert(p.input, data);
    }
    Ok(inputs)
}

/// A `[rows, cols]` row-major matrix transposed to `[cols, rows]`; `rows == 0` leaves the
/// data as it is.
fn turned<T: Copy + Default>(data: Vec<T>, rows: usize, cols: usize) -> Vec<T> {
    if rows == 0 {
        return data;
    }
    let mut out = vec![T::default(); data.len()];
    for (r, row) in data.chunks_exact(cols).enumerate() {
        for (c, &value) in row.iter().enumerate() {
            out[c * rows + r] = value;
        }
    }
    out
}

/// Read one tensor of a safetensors file as f32, checking its shape.
pub fn read_f32(path: &Path, name: &str, shape: &[usize]) -> anyhow::Result<Vec<f32>> {
    let file = File::open(path)?;
    // SAFETY: model files are immutable and stay mapped throughout decoding.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    let tensor = tensors.tensor(name)?;
    anyhow::ensure!(tensor.shape() == shape, "shape mismatch for {name}");
    Ok(tensor_to_f32(&tensor))
}

/// Whether tensors `a` and `b` of a safetensors file hold the same values — tied weights
/// saved twice. A missing `b` counts as tied.
pub fn same_tensor(path: &Path, a: &str, b: &str) -> anyhow::Result<bool> {
    let file = File::open(path)?;
    // SAFETY: model files are immutable and stay mapped throughout the comparison.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    let a = tensors.tensor(a)?;
    Ok(match tensors.tensor(b) {
        Ok(b) => a.dtype() == b.dtype() && a.shape() == b.shape() && a.data() == b.data(),
        Err(_) => true,
    })
}

/// Downloads model files from HuggingFace and returns the cache directory path.
pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();
    // Repos split across several files have an index; single-file repos do not. Only a
    // missing index means "single file": a failed weight download must surface as itself.
    let index_path = match repo.get("model.safetensors.index.json") {
        Ok(path) => path,
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

/// Download a DSpark drafter checkpoint; returns the paths of its `config.json` and
/// `model.safetensors`. The file is used as published, with no preparation step.
pub fn download_drafter(repo_id: &str) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let repo = Api::new()?.model(repo_id.to_string());
    Ok((repo.get("config.json")?, repo.get("model.safetensors")?))
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

    /// Write a fake model split across two files (F32 and BF16 sources, one norm weight).
    fn fake_model(dir: &Path) {
        let w0 = f32_bytes(&[1.0, -2.5, 0.125, 3.0]);
        let norm = f32_bytes(&[0.5, 1.5]);
        let w1: Vec<u8> = [4.0f32, -8.0]
            .iter()
            .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
            .collect();
        let view = |dtype, shape: &[usize], data| TensorView::new(dtype, shape.to_vec(), data);
        let a = [
            (
                "model.layers.0.mlp.up_proj.weight",
                view(Dtype::F32, &[2, 2], &w0).unwrap(),
            ),
            (
                "model.layers.0.input_layernorm.weight",
                view(Dtype::F32, &[2], &norm).unwrap(),
            ),
        ];
        let b = [(
            "model.layers.1.mlp.up_proj.weight",
            view(Dtype::BF16, &[2], &w1).unwrap(),
        )];
        safetensors::serialize_to_file(a, None, &dir.join("model-00001-of-00002.safetensors"))
            .unwrap();
        safetensors::serialize_to_file(b, None, &dir.join("model-00002-of-00002.safetensors"))
            .unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {
                "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.mlp.up_proj.weight": "model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
    }

    #[test]
    fn combine_converts_dtypes_and_leaves_no_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        fake_model(dir.path());
        let out = combine_safetensors_to_bf16(dir.path()).unwrap();
        assert_eq!(out, dir.path().join(COMBINED_WEIGHTS_FILE));

        let bytes = std::fs::read(&out).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(st.names().len(), 3);
        // F32 matrix → BF16 (these values are exactly representable).
        let up0 = st.tensor("model.layers.0.mlp.up_proj.weight").unwrap();
        assert_eq!((up0.dtype(), up0.shape()), (Dtype::BF16, &[2usize, 2][..]));
        assert_eq!(tensor_to_f32(&up0), vec![1.0, -2.5, 0.125, 3.0]);
        // Norm weights stay F32.
        let norm = st.tensor("model.layers.0.input_layernorm.weight").unwrap();
        assert_eq!(norm.dtype(), Dtype::F32);
        assert_eq!(tensor_to_f32(&norm), vec![0.5, 1.5]);
        // BF16 passes through untouched.
        let up1 = st.tensor("model.layers.1.mlp.up_proj.weight").unwrap();
        assert_eq!(up1.dtype(), Dtype::BF16);
        assert_eq!(tensor_to_f32(&up1), vec![4.0, -8.0]);

        let files = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(
            files, 4,
            "two sources, the index and the prepared file — no leftovers"
        );
    }

    #[test]
    fn interrupted_prepared_file_is_rebuilt_and_a_complete_one_is_reused() {
        let dir = tempfile::tempdir().unwrap();
        fake_model(dir.path());
        let out = combine_safetensors_to_bf16(dir.path()).unwrap();
        let good = std::fs::read(&out).unwrap();

        // What an interrupted write under the final name would have left.
        std::fs::write(&out, &good[..good.len() - 3]).unwrap();
        assert!(!valid_combined_file(&out));
        combine_safetensors_to_bf16(dir.path()).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), good);

        let before = std::fs::metadata(&out).unwrap().modified().unwrap();
        combine_safetensors_to_bf16(dir.path()).unwrap();
        assert_eq!(std::fs::metadata(&out).unwrap().modified().unwrap(), before);
    }
}

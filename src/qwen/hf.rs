use half::{bf16, f16};
use hf_hub::api::sync::Api;
use memmap2::MmapOptions;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

/// Index file structure for sharded safetensors models
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Stored tensor data with shape, dtype, and serialized bytes.
struct StoredTensor {
    shape: Vec<usize>,
    dtype: Dtype,
    data: Vec<u8>,
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

fn stored_tensor_bf16(name: &str, tensor: &safetensors::tensor::TensorView) -> StoredTensor {
    let shape = tensor.shape().to_vec();
    if keep_f32_in_bf16_pipeline(name) {
        StoredTensor {
            shape,
            dtype: Dtype::F32,
            data: tensor_to_f32_bytes(tensor),
        }
    } else {
        StoredTensor {
            shape,
            dtype: Dtype::BF16,
            data: tensor_to_bf16_bytes(tensor),
        }
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

/// Combines sharded safetensors into a single bf16 file (norm weights kept
/// F32). Linear projections, the token embedding and the lm_head are bf16; the
/// graph explicitly converts checkpoint storage to F32 arithmetic.
///
/// Qwen3-4B ties the lm_head to the embedding table, so no separate
/// `lm_head.weight` is stored — the model reuses `model.embed_tokens.weight`.
pub fn combine_safetensors_to_bf16(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined_bf16_v1.safetensors");

    if valid_combined_file(&output_path) {
        let existing_bytes = std::fs::metadata(&output_path)?.len();
        println!(
            "Using existing combined BF16 model at {} ({})",
            output_path.display(),
            format_bytes(existing_bytes)
        );
        return Ok(output_path);
    }

    let shard_files = model_shard_files(model_dir)?;
    println!(
        "Loading {} shard files (converting to BF16, norms F32)...",
        shard_files.len()
    );

    let mut all_tensors: HashMap<String, StoredTensor> = HashMap::new();

    for shard_path in &shard_files {
        println!(
            "  Loading {}...",
            shard_path.file_name().unwrap().to_string_lossy()
        );
        let file = File::open(shard_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        for name in st.names() {
            let tensor = st.tensor(name)?;
            all_tensors.insert(name.to_string(), stored_tensor_bf16(name, &tensor));
        }
    }

    println!("Extracted {} tensors", all_tensors.len());
    println!("Saving combined BF16 model to {}...", output_path.display());

    let tensor_views: HashMap<String, TensorView<'_>> = all_tensors
        .iter()
        .map(|(name, stored)| {
            let view = TensorView::new(stored.dtype, stored.shape.clone(), &stored.data).unwrap();
            (name.clone(), view)
        })
        .collect();

    let serialized = safetensors::serialize(&tensor_views, None)?;

    publish_atomic(&output_path, &serialized)?;

    println!("Combined BF16 model saved successfully!");
    Ok(output_path)
}

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

fn publish_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut temp = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    temp.persist(path).map_err(|e| e.error)?;
    Ok(())
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
        let data = match p.dtype {
            DType::Bf16 => {
                anyhow::ensure!(tensor.dtype() == Dtype::BF16, "expected BF16 {}", p.name);
                TensorData::BF16(
                    tensor
                        .data()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        .collect(),
                )
            }
            DType::F32 => TensorData::F32(tensor_to_f32(&tensor)),
            _ => anyhow::bail!("unsupported parameter dtype"),
        };
        inputs.insert(p.input, data);
    }
    Ok(inputs)
}

/// Downloads model files from HuggingFace and returns the cache directory path.
pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();
    // Try to download single shard model first
    if repo.get("model.safetensors").is_ok() {
        return Ok(model_dir);
    }
    // Otherwise download sharded model
    let index_path = repo.get("model.safetensors.index.json")?;
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
/// - model_combined_bf16_v1.safetensors
pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors_to_bf16(&model_dir)?;
    Ok(model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn atomic_publish_replaces_corrupt_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("combined.safetensors");
        std::fs::write(&path, b"interrupted").unwrap();
        assert!(!valid_combined_file(&path));
        let data = 1f32.to_le_bytes();
        let view = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        let bytes = safetensors::serialize([("weight", view)], None).unwrap();
        publish_atomic(&path, &bytes).unwrap();
        assert!(valid_combined_file(&path));
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }
}

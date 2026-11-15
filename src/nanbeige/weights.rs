//! Checkpoints: download from Hugging Face and read tensors straight from the published
//! safetensors files. Nothing is converted or rewritten on disk: the kernels read bf16 and
//! 4-bit (NVFP4) matrices in the layout the repositories store them.

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use anyhow::{anyhow, bail, ensure, Context, Result};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::Deserialize;

/// Memory-mapped safetensors files, addressed by tensor name.
pub struct Checkpoint {
    maps: Vec<Mmap>,
    /// Tensor name → index into `maps`.
    index: HashMap<String, usize>,
}

#[derive(Deserialize)]
struct RepoIndex {
    weight_map: HashMap<String, String>,
}

impl Checkpoint {
    /// Open local safetensors files.
    pub fn open(files: &[PathBuf]) -> Result<Self> {
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        for path in files {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            // SAFETY: model files are not modified while this process reads them.
            let map = unsafe { MmapOptions::new().map(&file)? };
            let names: Vec<String> = SafeTensors::deserialize(&map)
                .with_context(|| format!("parse {}", path.display()))?
                .names()
                .into_iter()
                .map(String::from)
                .collect();
            for name in names {
                index.insert(name, maps.len());
            }
            maps.push(map);
        }
        Ok(Self { maps, index })
    }

    /// Download the files of `repo` that hold the tensors `names` and open them. A repository
    /// split over several files publishes an index; only the files it names for these tensors
    /// are fetched. Files already in the Hugging Face cache are reused.
    pub fn download(repo: &str, names: &[String]) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context("hf-hub api")?;
        let repo_api = api.model(repo.to_string());
        let files: Vec<String> = match repo_api.get("model.safetensors.index.json") {
            Ok(path) => {
                let index: RepoIndex = serde_json::from_str(&std::fs::read_to_string(path)?)
                    .context("parse model.safetensors.index.json")?;
                let mut files = Vec::new();
                for name in names {
                    let file = index
                        .weight_map
                        .get(name)
                        .ok_or_else(|| anyhow!("{repo} has no tensor {name}"))?;
                    if !files.contains(file) {
                        files.push(file.clone());
                    }
                }
                files.sort();
                files
            }
            // Only a missing index means "one file"; a failed weight download surfaces below.
            Err(_) => vec!["model.safetensors".to_string()],
        };
        let mut paths = Vec::new();
        for (i, file) in files.iter().enumerate() {
            tracing::info!("{repo}: {file} ({} of {})", i + 1, files.len());
            paths.push(
                repo_api
                    .get(file)
                    .with_context(|| format!("download {file} from {repo}"))?,
            );
        }
        Self::open(&paths)
    }

    /// Whether the checkpoint holds a tensor of this name.
    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn view<R>(&self, name: &str, shape: &[usize], f: impl FnOnce(TensorView) -> R) -> Result<R> {
        let &map = self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("checkpoint has no tensor {name}"))?;
        let tensors = SafeTensors::deserialize(&self.maps[map])?;
        let tensor = tensors.tensor(name)?;
        ensure!(
            tensor.shape() == shape,
            "{name} has shape {:?}, expected {shape:?}",
            tensor.shape()
        );
        Ok(f(tensor))
    }

    /// A floating-point tensor.
    fn with<R>(&self, name: &str, shape: &[usize], f: impl FnOnce(TensorView) -> R) -> Result<R> {
        self.view(name, shape, |tensor| {
            ensure!(
                matches!(tensor.dtype(), Dtype::BF16 | Dtype::F16 | Dtype::F32),
                "{name} has unsupported dtype {:?}",
                tensor.dtype()
            );
            Ok(f(tensor))
        })?
    }

    /// The bytes of a tensor stored as `dtype`, straight from the file.
    fn raw<R>(
        &self,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R> {
        self.view(name, shape, |tensor| {
            ensure!(
                tensor.dtype() == dtype,
                "{name} has dtype {:?}, expected {dtype:?}",
                tensor.dtype()
            );
            Ok(f(tensor.data()))
        })?
    }

    /// Whether the matrix `name` is stored in 4 bits (NVFP4) rather than as floats.
    pub fn is_fp4(&self, name: &str) -> bool {
        self.has(&format!("{name}_scale"))
    }

    /// A 4-bit matrix of `[n_out, n_in]` weights, handed to `f` as it sits in the file:
    /// `[n_out, n_in / 2]` bytes of two 4-bit floats each (low nibble first),
    /// `[n_out, n_in / 16]` 8-bit float scales, and the matrix's f32 scale. A weight is
    /// the product of the three.
    pub fn fp4<R>(
        &self,
        name: &str,
        [n_out, n_in]: [usize; 2],
        f: impl FnOnce(&[u8], &[u8], f32) -> R,
    ) -> Result<R> {
        ensure!(
            n_in.is_multiple_of(FP4_GROUP),
            "{name}: 4-bit rows come in groups of {FP4_GROUP}"
        );
        let global = self.raw(&format!("{name}_scale_2"), Dtype::F32, &[1], |b| {
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })?;
        ensure!(
            global.is_finite() && global > 0.0,
            "{name}_scale_2 is {global}"
        );
        self.raw(name, Dtype::U8, &[n_out, n_in / 2], |w| {
            self.raw(
                &format!("{name}_scale"),
                Dtype::F8_E4M3,
                &[n_out, n_in / FP4_GROUP],
                |scale| f(w, scale, global),
            )
        })?
    }

    /// A tensor as bf16 bits: the form matrices take on the device.
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        self.with(name, shape, |t| match t.dtype() {
            Dtype::BF16 => t
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
            _ => to_f32(&t)
                .into_iter()
                .map(|x| bf16::from_f32(x).to_bits())
                .collect(),
        })
    }

    /// Two matrices of the same shape, one after the other: `[2 * shape[0], shape[1]]`.
    pub fn bf16_stacked(&self, a: &str, b: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let mut out = self.bf16(a, shape)?;
        out.extend(self.bf16(b, shape)?);
        Ok(out)
    }
    /// A tensor as f32: norm weights and the small tensors the host computes with.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.with(name, shape, |t| to_f32(&t))
    }
}

fn to_f32(tensor: &TensorView) -> Vec<f32> {
    let data = tensor.data();
    match tensor.dtype() {
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
        other => unreachable!("dtype {other:?} is rejected before conversion"),
    }
}

/// Weights per 8-bit scale in a 4-bit matrix.
pub const FP4_GROUP: usize = 16;

/// Magnitudes a 4-bit float takes, by its low three bits; the fourth is the sign.
#[cfg(test)]
const FP4_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The value of an 8-bit float: sign, 4-bit exponent biased by 7, 3-bit mantissa.
#[cfg(test)]
pub fn fp8_value(b: u8) -> f32 {
    let (e, m) = (((b >> 3) & 15) as i32, (b & 7) as f32);
    let v = if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

/// A 4-bit matrix expanded to f32, on the host: what the kernels are checked against.
#[cfg(test)]
pub fn fp4_dense(w: &[u8], scale: &[u8], global: f32) -> Vec<f32> {
    w.iter()
        .enumerate()
        .flat_map(|(i, &byte)| {
            let s = fp8_value(scale[i / (FP4_GROUP / 2)]) * global;
            [byte & 15, byte >> 4].map(|n| {
                let v = FP4_VALUES[(n & 7) as usize] * s;
                if n & 8 != 0 {
                    -v
                } else {
                    v
                }
            })
        })
        .collect()
}

/// Round a matrix to 4 bits the way the published checkpoint is laid out: the largest weight
/// of the matrix maps to the top of both scales, each group of 16 gets the nearest 8-bit
/// scale, each weight the nearest 4-bit value. Returns packed weights, scales and the
/// matrix's scale.
#[cfg(test)]
pub fn fp4_quantize(values: &[f32]) -> (Vec<u8>, Vec<u8>, f32) {
    assert!(values.len().is_multiple_of(FP4_GROUP));
    let nearest = |candidates: &mut dyn Iterator<Item = (u8, f32)>, x: f32| -> u8 {
        candidates
            .min_by(|a, b| (a.1 - x).abs().total_cmp(&(b.1 - x).abs()))
            .unwrap()
            .0
    };
    let amax = values.iter().fold(f32::MIN_POSITIVE, |m, v| m.max(v.abs()));
    let global = amax / (448.0 * 6.0);
    let (mut packed, mut scales) = (Vec::new(), Vec::new());
    for group in values.chunks(FP4_GROUP) {
        let gmax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        let code = nearest(
            &mut (1u8..0x7F).map(|c| (c, fp8_value(c))),
            gmax / 6.0 / global,
        );
        scales.push(code);
        let s = fp8_value(code) * global;
        let nibble = |v: f32| {
            let n = nearest(
                &mut FP4_VALUES.iter().enumerate().map(|(i, &q)| (i as u8, q)),
                v.abs() / s,
            );
            if v < 0.0 {
                n | 8
            } else {
                n
            }
        };
        packed.extend(group.chunks(2).map(|p| nibble(p[0]) | nibble(p[1]) << 4));
    }
    (packed, scales, global)
}

#[cfg(test)]
impl Checkpoint {
    /// A matrix as f32 whichever way it is stored.
    pub fn dense(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        if self.is_fp4(name) {
            self.fp4(name, [shape[0], shape[1]], fp4_dense)
        } else {
            self.f32(name, shape)
        }
    }
}

/// A tensor for a generated checkpoint: name, dtype, shape, bytes.
#[cfg(test)]
pub type RawTensor = (String, Dtype, Vec<usize>, Vec<u8>);

/// Write a safetensors file: generated models for tests.
#[cfg(test)]
pub fn write_tensors(path: &std::path::Path, tensors: &[RawTensor]) -> Result<()> {
    let views: Vec<(&str, TensorView)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            Ok((name.as_str(), TensorView::new(*dtype, shape.clone(), data)?))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(views, None, path)?;
    Ok(())
}

#[cfg(test)]
fn bf16_tensor(name: &str, shape: &[usize], values: &[f32]) -> RawTensor {
    let bytes = values
        .iter()
        .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
        .collect();
    (name.to_string(), Dtype::BF16, shape.to_vec(), bytes)
}

/// The three tensors of a 4-bit matrix, named as the published checkpoint names them.
#[cfg(test)]
pub fn fp4_tensors(
    name: &str,
    [n_out, n_in]: [usize; 2],
    packed: Vec<u8>,
    scales: Vec<u8>,
    global: f32,
) -> [RawTensor; 3] {
    [
        (name.to_string(), Dtype::U8, vec![n_out, n_in / 2], packed),
        (
            format!("{name}_scale"),
            Dtype::F8_E4M3,
            vec![n_out, n_in / FP4_GROUP],
            scales,
        ),
        (
            format!("{name}_scale_2"),
            Dtype::F32,
            vec![1],
            global.to_le_bytes().to_vec(),
        ),
    ]
}

/// Write `tensors` as bf16.
#[cfg(test)]
pub fn write_bf16(
    path: &std::path::Path,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<()> {
    write_fp4(path, tensors, |_| false)
}

/// Write `tensors` with the matrices `quantized` picks rounded to 4 bits and the rest as bf16.
#[cfg(test)]
pub fn write_fp4(
    path: &std::path::Path,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    quantized: impl Fn(&str) -> bool,
) -> Result<()> {
    let mut raw = Vec::new();
    for (name, shape, values) in tensors {
        if quantized(name) {
            let (packed, scales, global) = fp4_quantize(values);
            raw.extend(fp4_tensors(
                name,
                [shape[0], shape[1]],
                packed,
                scales,
                global,
            ));
        } else {
            raw.push(bf16_tensor(name, shape, values));
        }
    }
    write_tensors(path, &raw)
}

/// Fail early, with the file named, when a checkpoint does not match a model description.
pub fn check(checkpoint: &Checkpoint, parameters: &[(String, Vec<usize>)]) -> Result<()> {
    for (name, shape) in parameters {
        checkpoint.with(name, shape, |_| ())?;
    }
    if parameters.is_empty() {
        bail!("no parameters to load");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_tensors_by_name_across_files_and_checks_shapes() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (
            dir.path().join("a.safetensors"),
            dir.path().join("b.safetensors"),
        );
        write_bf16(&a, &[("w".into(), vec![2, 2], vec![1.0, -2.5, 0.125, 3.0])]).unwrap();
        write_bf16(&b, &[("n".into(), vec![2], vec![0.5, 1.5])]).unwrap();
        let ck = Checkpoint::open(&[a, b]).unwrap();
        assert_eq!(ck.f32("w", &[2, 2]).unwrap(), vec![1.0, -2.5, 0.125, 3.0]);
        assert_eq!(
            ck.bf16("n", &[2]).unwrap(),
            vec![bf16::from_f32(0.5).to_bits(), bf16::from_f32(1.5).to_bits()]
        );
        assert!(ck.f32("w", &[4]).is_err(), "wrong shape");
        assert!(ck.f32("missing", &[1]).is_err());
        check(&ck, &[("w".into(), vec![2, 2]), ("n".into(), vec![2])]).unwrap();
        assert!(check(&ck, &[("n".into(), vec![3])]).is_err());
    }

    #[test]
    fn eight_bit_floats_decode() {
        // Smallest subnormal, one, the largest value, a negative.
        assert_eq!(fp8_value(0x01), 2f32.powi(-9));
        assert_eq!(fp8_value(0x38), 1.0);
        assert_eq!(fp8_value(0x7E), 448.0);
        assert_eq!(fp8_value(0xB9), -1.125);
    }

    #[test]
    fn four_bit_matrices_round_trip_through_a_file() {
        // Two rows of 32: every 4-bit value and sign appears, at two scales per row.
        let exact: Vec<f32> = (0..64)
            .map(|i| {
                let v = FP4_VALUES[i % 8] * if i % 16 < 8 { 1.0 } else { -1.0 };
                v * if (i / 16) % 2 == 0 { 0.25 } else { 0.03125 }
            })
            .collect();
        let (packed, scales, global) = fp4_quantize(&exact);
        assert_eq!((packed.len(), scales.len()), (32, 4));
        // Low nibble first: weights 0 and 1 are 0 and 0.5 units.
        assert_eq!(packed[0], 0x10);
        let back = fp4_dense(&packed, &scales, global);
        for (a, b) in back.iter().zip(&exact) {
            assert!((a - b).abs() <= 1e-6 * b.abs(), "{a} vs {b}");
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q.safetensors");
        let tensors = [("m.weight".to_string(), vec![2, 32], exact)];
        write_fp4(&path, &tensors, |_| true).unwrap();
        let ck = Checkpoint::open(&[path]).unwrap();
        assert!(ck.is_fp4("m.weight") && !ck.is_fp4("m"));
        assert_eq!(ck.dense("m.weight", &[2, 32]).unwrap(), back);
        assert!(
            ck.fp4("m.weight", [4, 16], |_, _, _| ()).is_err(),
            "wrong shape"
        );
        assert!(ck.bf16("m.weight", &[2, 16]).is_err(), "not a float tensor");

        // Rounding arbitrary weights stays within half a step of the coarsest spacing.
        let rough: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 64) as f32 - 31.5) / 40.0)
            .collect();
        let (p, s, g) = fp4_quantize(&rough);
        let err = fp4_dense(&p, &s, g)
            .iter()
            .zip(&rough)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 0.15, "rounding error {err}");
    }
}

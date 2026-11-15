//! Compile the CUDA kernels in `cu/` to PTX with `nvcc` and leave them in `OUT_DIR`, where
//! `src/gpu.rs` embeds them. PTX is compiled for the GPU it runs on by the driver when the
//! module is loaded, so one build serves every card from `POTPORI_SM` up.
//!
//! Environment:
//! - `NVCC`: the compiler (default: `nvcc` on `PATH`, else `$CUDA_PATH/bin/nvcc`).
//! - `NVCC_CCBIN`: the host C++ compiler nvcc preprocesses with. Needed on Windows when
//!   `cl.exe` is not on `PATH`.
//! - `POTPORI_SM`: the oldest architecture to support (default `sm_75`, RTX 20 series).

use std::{env, path::PathBuf, process::Command};

fn main() {
    for var in ["NVCC", "NVCC_CCBIN", "POTPORI_SM", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=cu");

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        let in_toolkit = env::var_os("CUDA_PATH").map(|p| PathBuf::from(p).join("bin/nvcc"));
        match in_toolkit {
            Some(p) if p.exists() || p.with_extension("exe").exists() => {
                p.to_string_lossy().into_owned()
            }
            _ => "nvcc".into(),
        }
    });
    let arch = env::var("POTPORI_SM").unwrap_or_else(|_| "sm_75".into());

    let mut cmd = Command::new(&nvcc);
    cmd.args(["-ptx", "-O3", "-std=c++17", "-Icu"])
        .arg(format!("-arch={arch}"));
    if let Ok(ccbin) = env::var("NVCC_CCBIN") {
        cmd.arg("-ccbin").arg(ccbin);
    }
    cmd.arg("cu/kernels.cu")
        .arg("-o")
        .arg(out.join("kernels.ptx"));
    let status = cmd.status().unwrap_or_else(|e| {
        panic!("could not run {nvcc} ({e}): install the CUDA toolkit or set NVCC")
    });
    assert!(
        status.success(),
        "nvcc failed on cu/kernels.cu (on Windows, set NVCC_CCBIN to cl.exe)"
    );
}

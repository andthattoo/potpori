//! Luminal backend adapters (CUDA, Metal, and the CPU reference executor for tests) behind one
//! small trait. A runtime is created empty
//! with `Default`, compiled once for a shard's graph, then stepped; device allocations
//! outlive the compiled programs.
use super::model::{Inputs, QwenGraph};
use anyhow::Result;
use std::time::Duration;

/// Where a runtime's step time went, summed over `steps` calls (backends that measure it).
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)] // read by the Metal profiling test
pub struct StepProfile {
    pub steps: u64,
    /// Writing the step's inputs to the device.
    pub upload: Duration,
    /// Running the kernels, start to finish.
    pub execute: Duration,
    /// Reading the output back.
    pub readback: Duration,
    /// Program executions (one command buffer each).
    pub launches: u64,
}

pub trait QwenRuntime {
    /// Search and compile kernels for `graph`, and upload `weights`.
    fn compile(&mut self, graph: &QwenGraph, weights: Inputs, search_graphs: usize) -> Result<()>;
    /// Run `query` new positions with `context` positions visible in total; returns the
    /// graph's output (residual stream or logits) as f32.
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>>;
    /// Clear the KV cache.
    fn reset(&mut self) -> Result<()>;
    /// Time and launch counts since compilation, if this backend measures them.
    #[allow(dead_code)] // called by the Metal profiling test
    fn profile(&self) -> Option<StepProfile> {
        None
    }
    /// A summary of the compiled programs — kernels by kind, copies, largest buffers — for
    /// diagnosing where time goes.
    #[allow(dead_code)] // called by the Metal profiling test
    fn describe(&self) -> Option<String> {
        None
    }
}

// The adapters keep the per-bucket search reports of the Luminal examples they derive from;
// nothing reads them yet.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod cuda;
#[cfg(feature = "cuda")]
mod cuda_memory;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(dead_code)]
mod metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_memory;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(unused_imports)] // used by the Metal profiling tests
pub use metal::kernel_kinds as metal_kernel_kinds;
#[cfg(feature = "reference")]
mod reference;
#[cfg(feature = "reference")]
#[cfg_attr(not(test), allow(unused_imports))]
pub use reference::CpuRuntime;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
macro_rules! adapter {
    ($name:ident, $module:ident, $backend:ident, $native:ident) => {
        #[derive(Default)]
        pub struct $name {
            backend: Option<$module::$backend>,
        }
        impl QwenRuntime for $name {
            fn compile(
                &mut self,
                graph: &QwenGraph,
                weights: Inputs,
                search_graphs: usize,
            ) -> Result<()> {
                let options = $native::CompileOptions {
                    generations: search_graphs.div_ceil(8).max(1),
                    ..Default::default()
                };
                self.backend = Some($module::$backend::compile(graph, weights, &options)?);
                Ok(())
            }
            fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
                self.backend
                    .as_mut()
                    .expect("compile before execution")
                    .step(inputs, query, context)
            }
            fn reset(&mut self) -> Result<()> {
                self.backend.as_mut().expect("compile before reset").reset()
            }
            fn profile(&self) -> Option<StepProfile> {
                self.backend.as_ref().and_then(|backend| backend.profile())
            }
            fn describe(&self) -> Option<String> {
                self.backend.as_ref().and_then(|backend| backend.describe())
            }
        }
    };
}
#[cfg(feature = "cuda")]
adapter!(CudaRuntime, cuda, CudaBackend, luminal_cuda_lite);
#[cfg(all(feature = "metal", target_os = "macos"))]
adapter!(MetalRuntime, metal, MetalBackend, luminal_metal);

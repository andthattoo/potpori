//! Native Luminal backend adapters. Device allocations outlive compiled programs.
use super::model::{Inputs, QwenGraph};
use anyhow::Result;
pub trait QwenRuntime {
    fn compile(&mut self, graph: &QwenGraph, weights: Inputs, search_graphs: usize) -> Result<()>;
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>>;
    fn reset(&mut self) -> Result<()>;
}
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod cuda_memory;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_memory;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
macro_rules! adapter {
    ($name:ident, $module:ident, $backend:ident, $native:ident) => {
        #[derive(Default)]
        pub struct $name {
            backend: Option<$module::$backend>,
        }
        impl $name {
            pub fn new() -> Result<Self> {
                Ok(Self::default())
            }
            pub fn initialize(_: ()) -> Self {
                Self::default()
            }
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
        }
    };
}
#[cfg(feature = "cuda")]
adapter!(CudaRuntime, cuda, CudaBackend, luminal_cuda_lite);
#[cfg(all(feature = "metal", target_os = "macos"))]
adapter!(MetalRuntime, metal, MetalBackend, luminal_metal);

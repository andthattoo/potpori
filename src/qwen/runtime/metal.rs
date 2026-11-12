// Adapted from Luminal llm_chat backend at 1cbb917 (MIT/Apache-2.0).
//! The Metal runtime's boundary statement for this example's chat graph,
//! and the execution loop over it.
use crate::qwen::model::{Inputs, QwenGraph as LlmGraph, StateBinding, TensorData};
use anyhow::{ensure, Result};
use luminal::prelude::*;
use luminal::{
    bucketing::BucketSet,
    dtype::PlanDtype,
    layout_ir::{Access, FreedBy},
    memory::{PersistentBinding, ProgramMemory, ResourceId, SharedArenaPlan},
};
use luminal_metal::device::ExternalBuffer;
use luminal_metal::{
    bindings::MetalBindings, metal_registry, CompileOptions, HostBuffer, MetalRuntime,
};

/// THE BINDING IS THE STATEMENT. The chat graph says nothing about
/// boundaries; this says all of it:
///
/// - Parameters, RoPE pairing matrices and KV state use external storage,
///   owned and initialized by the application.
/// - Tokens, positions, the gather/scatter index maps, the last-row index
///   and the RoPE tables are explicitly uploaded by `step` before execution.
///   The application reads logits after execution for CPU sampling.
/// - Each KV output is bound ON its input's buffer, which is the one
///   spelling of "this step's cache writes last step's storage". The
///   buffer's contents permission has to say so, hence the re-declaration.
/// - The logits are the only value read back.
pub fn bindings(graph: &LlmGraph) -> MetalBindings {
    let mut bindings = MetalBindings::new();
    for parameter in &graph.parameters {
        bindings.input_external(parameter.input);
    }
    for matrix in graph.rope_matrices() {
        bindings.input_external(matrix);
    }
    for state in &graph.state {
        let home = bindings.input_external(state.input);
        bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
        bindings.output_on(state.output, home);
    }
    for input in graph.step_input_ids() {
        bindings.input(input);
    }
    bindings.output(graph.logits);
    bindings
}

/// Luminal's Metal kernels, minus any named in `POTPORI_METAL_WITHOUT` (comma-separated
/// labels such as `IndexMapApplyMaterialize`) — an experiment knob: the planner chooses per
/// site between kernels that read through a view and kernels that first copy their operand.
fn registry() -> Vec<luminal_metal::RegisteredOp> {
    let without = std::env::var("POTPORI_METAL_WITHOUT").unwrap_or_default();
    let dropped: Vec<&str> = without.split(',').filter(|s| !s.is_empty()).collect();
    luminal_metal::metal_registry_filtered(|op| !dropped.contains(&op.label()))
}

/// The labels of every kernel kind the planner can choose from.
pub fn kernel_kinds() -> Vec<&'static str> {
    metal_registry().iter().map(|op| op.label()).collect()
}

pub type BucketPlan =
    crate::qwen::model::BucketPlan<luminal_metal::layouts::MetalPlan, luminal_metal::SearchOutcome>;

pub struct MetalBackend {
    programs: BucketSet<(MetalRuntime, FxHashMap<i64, ExternalBuffer>)>,
    reports: Vec<BucketPlan>,
    state: Vec<StateBinding>,
    logits: NodeIndex,
    memory: SharedArenaPlan,
    profile: super::StepProfile,
    // Programs drop before the memory their executable addresses refer to.
    arena: super::metal_memory::Allocation,
}
impl MetalBackend {
    pub fn compile(
        graph: &LlmGraph,
        mut weights: Inputs,
        options: &CompileOptions,
    ) -> Result<Self> {
        let device = luminal_metal::device::MetalDevice::new()?;
        weights.extend(graph.initial_inputs());
        let mut data: FxHashMap<_, HostBuffer> =
            weights.into_iter().map(|(id, v)| (id, host(v))).collect();
        let resources: Vec<_> = data.keys().copied().collect();
        let mut programs = Vec::new();
        let mut reports = Vec::new();
        let mut requirements = Vec::new();
        for spec in crate::qwen::model::buckets(graph)? {
            let q = spec.profile_dims()[&'q'.into()];
            let c = spec.profile_dims()[&'c'.into()];
            for (id, value) in graph.profile_inputs(q, c)? {
                data.insert(id, host(value));
            }
            let mut runtime = MetalRuntime::load_with(&graph.graph, bindings(graph), registry())?
                .with_device(&device)?;
            let outcome = runtime.search(spec.bounds(), spec.profile_dims(), &data, options)?;
            let scratch_bytes = runtime.arena_bytes()?;
            let resource_bindings = resources
                .iter()
                .map(|id| {
                    Ok(PersistentBinding {
                        resource: ResourceId(id.index() as u64),
                        buffer: runtime.input_buffer(*id)?,
                        bytes: data[id].bytes.len(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            requirements.push(ProgramMemory {
                scratch_bytes,
                bindings: resource_bindings,
            });
            reports.push(BucketPlan {
                ranges: spec.bounds().ranges(),
                representative: spec.profile_dims().clone(),
                plan: outcome.best_plan.clone(),
                finalist_rank: outcome.finalist_rank,
                slab_bytes: scratch_bytes,
                outcome,
            });
            programs.push((spec, (runtime, FxHashMap::default())));
        }
        let memory = SharedArenaPlan::build(&requirements, usize::MAX)?;
        let mut arena =
            super::metal_memory::Allocation::new(device.device(), device.queue(), memory.bytes)?;
        for id in resources {
            let home = memory.homes[&ResourceId(id.index() as u64)];
            let buffer = &data[&id];
            ensure!(
                home.bytes == buffer.bytes.len(),
                "persistent input size mismatch"
            );
            arena.write(home.offset, &buffer.bytes)?;
        }
        for (index, (_, (_, external))) in programs.iter_mut().enumerate() {
            for (&buffer, home) in memory.bindings(index) {
                external.insert(
                    buffer,
                    luminal_metal::device::ExternalBuffer {
                        buffer: arena.buffer().clone(),
                        offset: home.offset,
                        bytes: home.bytes,
                    },
                );
            }
        }
        Ok(Self {
            programs: BucketSet::new(programs)?,
            reports,
            state: graph.state.clone(),
            logits: graph.logits,
            memory,
            profile: Default::default(),
            arena,
        })
    }

    pub fn bucket_plans(&self) -> &[BucketPlan] {
        &self.reports
    }

    pub fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
        let dims = [('q'.into(), query), ('c'.into(), context)]
            .into_iter()
            .collect();
        let (runtime, external) = self.programs.select_mut(&dims)?;
        runtime.set_dim('q', query);
        runtime.set_dim('c', context);
        let t0 = std::time::Instant::now();
        let mut uploads = Vec::with_capacity(inputs.len());
        for (id, value) in inputs {
            let data = host(value);
            let range = runtime.input_arena_range(id)?;
            ensure!(range.bytes == data.bytes.len(), "step input size mismatch");
            uploads.push((range.offset, data.bytes));
        }
        let parts: Vec<(usize, &[u8])> = uploads
            .iter()
            .map(|(offset, bytes)| (*offset, bytes.as_slice()))
            .collect();
        self.arena.write_all(&parts)?;
        let t1 = std::time::Instant::now();
        let launched = |runtime: &MetalRuntime| runtime.graph_stats().map_or(0, |s| s.launches);
        let before = launched(runtime);
        runtime.execute_external(self.arena.buffer(), external)?;
        let t2 = std::time::Instant::now();
        self.profile.launches += launched(runtime).saturating_sub(before);
        let range = runtime.output_arena_range(self.logits)?;
        let binding = runtime.output_layout(self.logits)?;
        let data = HostBuffer::new(PlanDtype::F32, self.arena.read(range.offset, range.bytes)?)?;
        let output = luminal_metal::layouts::dense_f32(&data.as_f32()?, &binding.layout);
        self.profile.steps += 1;
        self.profile.upload += t1 - t0;
        self.profile.execute += t2 - t1;
        self.profile.readback += t2.elapsed();
        output
    }

    pub fn profile(&self) -> Option<super::StepProfile> {
        Some(self.profile)
    }

    /// Per bucket: how many kernels of each kind the chosen plan launches and how many bytes
    /// they write at the profiled shape, the whole-buffer copies, and the largest buffers.
    pub fn describe(&self) -> Option<String> {
        use luminal::bufferize::BufferNode;
        use std::fmt::Write as _;
        let mib = |bytes: usize| bytes as f64 / (1 << 20) as f64;
        let mut out = String::new();
        for report in &self.reports {
            let dims = &report.representative;
            let size = |id: &luminal::bufferize::BufferId| {
                report
                    .plan
                    .buffers
                    .get(id)
                    .and_then(|b| luminal_metal::symbolic::bytes(&b.layout, dims).ok())
                    .unwrap_or(0)
            };
            let mut kinds: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
            let mut copies = (0usize, 0usize);
            for node in report.plan.dag.node_weights() {
                match node {
                    BufferNode::Compute { op, writes, .. } => {
                        let entry = kinds.entry(op.label().to_string()).or_default();
                        entry.0 += 1;
                        entry.1 += writes.iter().map(size).sum::<usize>();
                    }
                    BufferNode::BufferCopy { src, .. }
                    | BufferNode::BufferCopyValue { src, .. } => {
                        copies.0 += 1;
                        copies.1 += size(src);
                    }
                    _ => {}
                }
            }
            let kernels: usize = kinds.values().map(|k| k.0).sum();
            let written: usize = kinds.values().map(|k| k.1).sum();
            let _ = writeln!(
                out,
                "bucket {:?} profiled at {:?}: {kernels} compute nodes writing {:.1} MiB, {} \
                 copies of {:.1} MiB, scratch {:.1} MiB",
                report.ranges,
                dims,
                mib(written),
                copies.0,
                mib(copies.1),
                mib(report.slab_bytes),
            );
            let mut kinds: Vec<_> = kinds.into_iter().collect();
            kinds.sort_by_key(|(_, (count, bytes))| std::cmp::Reverse((*bytes, *count)));
            for (label, (count, bytes)) in kinds.iter().take(30) {
                let _ = writeln!(
                    out,
                    "  {count:5} x {label:<40} writes {:9.2} MiB",
                    mib(*bytes)
                );
            }
            let mut buffers: Vec<_> = report
                .plan
                .buffers
                .values()
                .map(|b| (size(&b.id), b.label.as_str()))
                .collect();
            buffers.sort_unstable_by_key(|b| std::cmp::Reverse(b.0));
            for (bytes, label) in buffers.iter().take(8) {
                let _ = writeln!(out, "  buffer {:9.2} MiB  {label}", mib(*bytes));
            }
        }
        Some(out)
    }

    pub fn reset(&mut self) -> Result<()> {
        for state in &self.state {
            let home = self.memory.homes[&ResourceId(state.input.index() as u64)];
            let buffer = host(TensorData::zeros(state.dtype, state.elements)?);
            ensure!(buffer.bytes.len() == home.bytes, "state size mismatch");
            self.arena.write(home.offset, &buffer.bytes)?;
        }
        Ok(())
    }
}

fn host(value: TensorData) -> HostBuffer {
    match value {
        TensorData::F32(v) => v.into(),
        TensorData::BF16(v) => HostBuffer {
            dtype: PlanDtype::Bf16,
            bytes: u16_bytes(&v),
        },
        TensorData::F16(v) => HostBuffer::new(PlanDtype::F16, u16_bytes(&v)).unwrap(),
        TensorData::I32(v) => v.into(),
    }
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

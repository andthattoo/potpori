//! Luminal's CPU reference executor behind [`QwenRuntime`]: exact and slow, for tests on tiny
//! models. One program covers every chunk size; the KV state is copied back by hand after
//! each step, where the GPU adapters bind it in place.
use crate::qwen::model::{Inputs, QwenGraph, StateBinding, TensorData};
use anyhow::{Context, Result};
use luminal::prelude::*;
use luminal::shape::SymbolBounds;
use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};

use super::QwenRuntime;

#[derive(Default)]
pub struct CpuRuntime {
    program: Option<Program>,
}

struct Program {
    runtime: ReferenceRuntime,
    state: Vec<StateBinding>,
    output: NodeIndex,
}

fn typed(value: TensorData) -> TypedBuffer {
    match value {
        TensorData::F32(v) => v.into(),
        TensorData::I32(v) => v.into(),
        TensorData::BF16(v) => v
            .into_iter()
            .map(half::bf16::from_bits)
            .collect::<Vec<_>>()
            .into(),
        TensorData::F16(v) => v
            .into_iter()
            .map(half::f16::from_bits)
            .collect::<Vec<_>>()
            .into(),
    }
}

impl QwenRuntime for CpuRuntime {
    fn compile(&mut self, graph: &QwenGraph, weights: Inputs, _search_graphs: usize) -> Result<()> {
        let specs = crate::qwen::model::buckets(graph)?;
        let rows = |pick: fn(&(usize, usize)) -> usize, fold: fn(usize, usize) -> usize| {
            specs
                .iter()
                .map(|spec| pick(&spec.bounds().ranges()[&'q'.into()]))
                .reduce(fold)
                .expect("at least one bucket")
        };
        let bounds = SymbolBounds::from_ranges([
            (
                'q'.into(),
                (rows(|r| r.0, usize::min), rows(|r| r.1, usize::max)),
            ),
            ('c'.into(), (1, graph.capacity)),
        ])?;
        let profile = specs.last().expect("at least one bucket").profile_dims();
        let (q, c) = (profile[&'q'.into()], profile[&'c'.into()]);

        let mut outputs = vec![graph.logits];
        outputs.extend(graph.state.iter().map(|s| s.output));
        let bindings = ReferenceBindings::dense(&graph.graph.logical, &outputs);
        let mut runtime = ReferenceRuntime::load_with(&graph.graph, bindings)?;

        let mut data: FxHashMap<NodeIndex, TypedBuffer> = FxHashMap::default();
        for (id, value) in weights.into_iter().chain(graph.initial_inputs()) {
            data.insert(id, typed(value));
        }
        let persistent: Vec<_> = data.keys().copied().collect();
        for (id, value) in graph.profile_inputs(q, c)? {
            data.insert(id, typed(value));
        }
        runtime
            .search(
                &bounds,
                profile,
                &data,
                &luminal_reference::harness_search_options(),
            )
            .context("reference search")?;
        for id in persistent {
            runtime.set_data(id, data.remove(&id).expect("collected above"));
        }
        self.program = Some(Program {
            runtime,
            state: graph.state.clone(),
            output: graph.logits,
        });
        Ok(())
    }

    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
        let program = self.program.as_mut().expect("compile before execution");
        program.runtime.set_dim('q', query);
        program.runtime.set_dim('c', context);
        for (id, value) in inputs {
            program.runtime.set_data(id, typed(value));
        }
        program.runtime.execute()?;
        for state in &program.state {
            let next = program.runtime.get_f32(state.output)?.clone();
            program.runtime.set_data(state.input, next);
        }
        Ok(program.runtime.get_f32(program.output)?.clone())
    }

    fn reset(&mut self) -> Result<()> {
        let program = self.program.as_mut().expect("compile before reset");
        for state in &program.state {
            program
                .runtime
                .set_data(state.input, vec![0f32; state.elements]);
        }
        Ok(())
    }
}

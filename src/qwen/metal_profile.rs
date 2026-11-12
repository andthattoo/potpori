//! Where a shard call's time goes on Metal. `#[ignore]` — needs the GPU and Qwen3-0.6B.
//!
//!   cargo test --release --features metal metal_profile -- --ignored --nocapture
//!
//! Three questions, each answered by numbers this prints:
//!   1. What does launching a kernel cost on this machine, with nothing in it?
//!   2. How does a call's time split into upload, kernels and readback, and how does it move
//!      with the number of layers, the chunk size, the visible context and the cache size?
//!   3. What is in the compiled plan: how many kernels, of which kinds, writing how much?

use std::time::Instant;

use metal::{Device, MTLResourceOptions, MTLSize};

use super::hf::{prepare_hf_model, COMBINED_WEIGHTS_FILE};
use super::model::QwenConfig;
use super::runtime::MetalRuntime;
use super::shard::Shard;
use crate::engine::TokenId;
use crate::node::ShardEngine;

const CFG: QwenConfig = QwenConfig::QWEN3_0_6B;
const CHUNK: usize = 8;
const SEARCH_GRAPHS: usize = 200;
const CALLS: usize = 20;

/// `passes` launches of a kernel that touches 256 floats, in one command buffer: each in
/// its own compute pass (as Luminal encodes them), or all in a single pass.
fn empty_launches(passes: usize, one_pass: bool) -> f64 {
    const SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
        kernel void k(device float* out, uint gid [[thread_position_in_grid]]) {\n\
            out[gid] = out[gid] + 1.0f;\n}\n";
    let device = Device::system_default().expect("a Metal device");
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_source(SOURCE, &metal::CompileOptions::new())
        .expect("compile the empty kernel");
    let function = library.get_function("k", None).expect("kernel function");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("pipeline");
    let buffer = device.new_buffer(4096, MTLResourceOptions::StorageModePrivate);
    let run = || {
        objc::rc::autoreleasepool(|| {
            let command = queue.new_command_buffer();
            let launch = |encoder: &metal::ComputeCommandEncoderRef| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&buffer), 0);
                encoder.dispatch_threads(MTLSize::new(256, 1, 1), MTLSize::new(256, 1, 1));
            };
            if one_pass {
                let encoder = command.new_compute_command_encoder();
                (0..passes).for_each(|_| launch(encoder));
                encoder.end_encoding();
            } else {
                for _ in 0..passes {
                    let encoder = command.new_compute_command_encoder();
                    launch(encoder);
                    encoder.end_encoding();
                }
            }
            command.commit();
            command.wait_until_completed();
        })
    };
    run(); // warm up
    let t0 = Instant::now();
    for _ in 0..5 {
        run();
    }
    t0.elapsed().as_secs_f64() * 1e3 / 5.0
}

fn load(range: (usize, usize), capacity: usize) -> Shard<MetalRuntime> {
    let dir = prepare_hf_model("Qwen/Qwen3-0.6B").expect("prepare 0.6B");
    Shard::load_chunked(
        MetalRuntime::default(),
        CFG,
        range,
        range.0 == 0,
        range.1 == CFG.layers,
        dir.join(COMBINED_WEIGHTS_FILE).to_str().unwrap(),
        capacity,
        SEARCH_GRAPHS,
        CHUNK,
    )
}

/// Mean milliseconds per call (total, upload, kernels, readback) for `rows` rows with
/// `context` positions visible.
fn time(shard: &mut Shard<MetalRuntime>, rows: usize, context: usize) -> [f64; 4] {
    let pos = (context - rows) as u32;
    let call = |shard: &mut Shard<MetalRuntime>| {
        if shard.range().0 == 0 {
            let tokens: Vec<TokenId> = (0..rows as TokenId).map(|i| 1000 + i).collect();
            shard.forward_tokens(pos, &tokens)
        } else {
            shard.forward_hidden(pos, rows, vec![0.01; rows * CFG.hidden])
        }
    };
    call(shard); // warm up
    let before = shard.profile().expect("the Metal backend measures");
    let t0 = Instant::now();
    for _ in 0..CALLS {
        call(shard);
    }
    let total = t0.elapsed().as_secs_f64() * 1e3 / CALLS as f64;
    let after = shard.profile().unwrap();
    let ms =
        |a: std::time::Duration, b: std::time::Duration| (a - b).as_secs_f64() * 1e3 / CALLS as f64;
    [
        total,
        ms(after.upload, before.upload),
        ms(after.execute, before.execute),
        ms(after.readback, before.readback),
    ]
}

/// What happens when the planner may not copy operands: for each kernel kind that
/// materializes a view, the same shards without it. A plan may not exist.
#[test]
#[ignore = "needs Metal and downloads Qwen3-0.6B"]
fn metal_without_copies() {
    println!("kernel kinds: {:?}", super::runtime::metal_kernel_kinds());
    let l = CFG.layers;
    for without in [
        "",
        "IndexMapApplyMaterialize",
        "MaterializeLayoutCopy",
        "IndexMapApplyMaterialize,MaterializeLayoutCopy",
    ] {
        std::env::set_var("POTPORI_METAL_WITHOUT", without);
        for (name, range) in [
            ("4 layers", (1, 5)),
            ("last layer + output head", (l - 1, l)),
        ] {
            let loaded = std::panic::catch_unwind(|| load(range, 512));
            let Ok(mut shard) = loaded else {
                println!("without [{without}] {name}: no plan");
                continue;
            };
            for (rows, context) in [(1, 16), (CHUNK, 16), (CHUNK, 500)] {
                let [total, _, kernels, _] = time(&mut shard, rows, context);
                println!(
                    "without [{without}] {name:26} rows {rows} context {context:3}: {total:7.2} \
                     ms ({kernels:7.2} in kernels)"
                );
            }
            let plan = shard.describe().unwrap_or_default();
            for line in plan.lines().filter(|l| l.starts_with("bucket")) {
                println!("    {line}");
            }
        }
    }
}

/// The same profile with 16-bit weight storage: half the memory, plus a cast per call.
#[test]
#[ignore = "needs Metal and downloads Qwen3-0.6B"]
fn metal_f16_profile() {
    std::env::set_var("POTPORI_WEIGHT_DTYPE", "f16");
    metal_profile();
}

#[test]
#[ignore = "needs Metal and downloads Qwen3-0.6B"]
fn metal_profile() {
    println!("\n== 1. Launching kernels that do nothing ==");
    for passes in [100, 1000] {
        println!(
            "{passes:5} launches: {:8.2} ms as separate passes, {:8.2} ms in one pass",
            empty_launches(passes, false),
            empty_launches(passes, true),
        );
    }

    println!("\n== 2. Shard calls (ms per call: total = upload + kernels + readback) ==");
    let l = CFG.layers;
    let cases: [(&str, (usize, usize)); 3] = [
        ("1 layer", (1, 2)),
        ("4 layers", (1, 5)),
        ("last layer + output head", (l - 1, l)),
    ];
    let mut plans = Vec::new();
    for (name, range) in cases {
        let mut shard = load(range, 512);
        for (rows, context) in [(1, 16), (CHUNK, 16), (1, 500), (CHUNK, 500)] {
            let [total, upload, kernels, readback] = time(&mut shard, rows, context);
            println!(
                "{name:26} rows {rows} context {context:3}: {total:7.2} = {upload:6.2} + \
                 {kernels:7.2} + {readback:6.2}"
            );
        }
        plans.push((name, shard.describe().unwrap_or_default()));
    }

    println!("\n== 3. Compiled plans ==");
    for (name, plan) in plans {
        println!("--- {name} ---\n{plan}");
    }
}

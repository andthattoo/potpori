# Historical design notes

These documents were written before the current implementation and are kept for the
reasoning behind it. They describe earlier plans — a llama.cpp-based design, a router-driven
network of nodes, GLM-5.2 as the target model — and benchmarks from an older Luminal revision.
Where they disagree with the code, the [README](../../README.md) or
[the migration notes](../luminal-migration.md), those win.

| Document | What it is |
|---|---|
| [pipeline-specdecode-design.md](pipeline-specdecode-design.md) | The original case for pipeline parallelism plus speculative decoding over WAN, with the latency math. |
| [shard-adoption-brief.md](shard-adoption-brief.md) | What was taken from the `shard` project: pipelined verification chunks and direct return. |
| [substrate-decision.md](substrate-decision.md) | Why Luminal was chosen over llama.cpp, Candle, mistral.rs and vLLM. |
| [chain-formation-design.md](chain-formation-design.md) | A control-plane design for forming and healing chains at network scale. Not implemented. |
| [cuda-substrate-runbook.md](cuda-substrate-runbook.md) | The first CUDA bring-up on a rented A6000, on the old Luminal revision. |

Ideas from these notes that the code does not have yet: direct return from the tail to the
head, NAT traversal, reconnection after a dropped node, an activation codec, sampling.

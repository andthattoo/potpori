# First WAN test — a Mac (Metal) and an NVIDIA box (CUDA) in one chain

Goal: one real generation across two machines and two backends, with numbers. This is the
first time the chain leaves loopback and the first time a CUDA shard is part of one, so expect
to learn something; the table at the bottom maps error messages to causes.

What the run prints, and what to write down (numbers here are made up):

```
[212 tokens in 21.4s = 9.9 tok/s overall, 11.6 tok/s after prefill | 2.71 tok/traversal over 78 traversals | γ=7 depth=4 chain=2 nodes]
[time: prefill 3.10s (41 prompt tokens) | head shard 4.20s | waiting on chain 9.80s | draft + rest 4.30s]
[link: RTT 38.2 ms at connect, 36.9 ms now | 85 frames, 80.0 KiB/frame | head cuda/Bf16 → metal/F32]
```

- **tok/traversal** is acceptance: how many tokens each chain round trip commits.
- **waiting on chain** vs **head shard** vs **draft + rest** says where the time goes.
- Run the same prompt with `--depth 1` and `--depth 4`: the difference is what pipelining buys
  over this link.

## Which machine is the head?

Two layouts work. Start with A.

| | A — NVIDIA box is the head | B — Mac is the head |
|---|---|---|
| Mac runs | `potpori serve` (tail) | `potpori chat` |
| NVIDIA box runs | `potpori chat` | `potpori serve` (tail) |
| Inbound connection to | the Mac | the NVIDIA box |
| Catch | you type the prompt on the NVIDIA box | WSL2 does not accept inbound UDP by default — needs mirrored networking and a firewall rule (below) |

The head dials the tail, so only the tail needs to be reachable. In layout A that is the Mac,
which needs nothing special.

## 0. Same code on both machines

Both nodes must be built from the same commit (the handshake refuses a protocol mismatch).

```bash
git fetch && git checkout fix/driver-and-config && git pull
```

## 1. Network: Tailscale on both

Install Tailscale on the Mac and on the **Windows host** (not inside WSL — that is Tailscale's
own guidance; running it in both breaks traffic). Then on the Mac:

```bash
tailscale ip -4          # the Mac's address, e.g. 100.x.y.z
```

From WSL on the NVIDIA box, check the path and the round trip:

```bash
ping -c 5 100.x.y.z      # the Mac
```

If both machines sit on the same LAN, Tailscale connects them directly and the RTT is about
1 ms — a fine first run, but not a WAN number. For a real one, put one machine on a different
network (a phone hotspot is enough).

Pick a shared secret for the session, e.g. `openssl rand -hex 16`, and use the same value as
`POTPORI_TOKEN` on both machines.

## 2. The Mac

```bash
cargo build --release --features metal
```

Layout A — the Mac is the tail (layers 18–36 of Qwen3-4B):

```bash
POTPORI_TOKEN=<secret> ./target/release/potpori serve \
  --listen 0.0.0.0:4010 --model Qwen/Qwen3-4B --layers 18:36 --role tail \
  --chunk 8 --max-seq 768
```

Wait for `shard ready — serving [18,36) as tail`. The first run downloads about 8 GB and runs
the kernel search (minutes). If macOS asks whether `potpori` may accept incoming connections,
allow it.

## 3. The NVIDIA box (Windows + WSL2)

Luminal's CUDA backend has only been run on Linux, so use WSL2 rather than a native Windows
build.

1. Windows: a current NVIDIA driver (the RTX 50 series needs a recent one), then
   `wsl --install -d Ubuntu-24.04`. **Do not install an NVIDIA driver inside WSL** — the
   Windows driver is passed through. `nvidia-smi` inside WSL should list the 5080.
2. Inside WSL, the CUDA **toolkit**: Luminal compiles kernels at runtime, so it needs the
   toolkit, and `luminal_cuda_lite` needs **12.8 or newer**. Install only the toolkit
   package — NVIDIA's WSL guide says not to use the `cuda` / `cuda-drivers` meta-packages
   there. The network-repo route (check NVIDIA's download page for WSL-Ubuntu if it differs):

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-8 build-essential pkg-config git curl
   echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
   echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc && nvcc --version      # must say 12.8 or newer
   ```
3. Rust and the build:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"
   git clone https://github.com/andthattoo/potpori && cd potpori
   git checkout fix/driver-and-config
   cargo build --release --features cuda
   ```
4. Preflight — proves the build, NVRTC and a correct forward on this GPU before any
   networking is involved (the check from `docs/cuda-substrate-runbook.md`):

   ```bash
   cargo test --release --features cuda cuda_substrate_check -- --ignored --nocapture
   ```

Layout A — this box is the head. Create `chain.toml`:

```toml
model = "Qwen/Qwen3-4B"
draft_model = "Qwen/Qwen3-0.6B"
gamma = 7
max_seq = 768
depth = 4

[[nodes]]                  # head — this machine
layers = [0, 18]

[[nodes]]                  # tail — the Mac
addr = "100.x.y.z:4010"    # the Mac's Tailscale IP
layers = [18, 36]
```

```bash
POTPORI_TOKEN=<secret> ./target/release/potpori chat --chain chain.toml --max-new 200 \
  --prompt "Explain speculative decoding in one paragraph."
```

Then the same command with `--depth 1`.

## Layout B — the Mac is the head

Swap the roles: the NVIDIA box runs the `serve` command from step 2 (in WSL; `--chunk 8` or
`--chunk 0`), the Mac runs `chat` with the NVIDIA box's Tailscale IP in `chain.toml`.

WSL2 sits behind NAT, so inbound UDP from the tailnet does not reach it by default. On
Windows 11 22H2 or newer, switch WSL to mirrored networking and allow inbound traffic. These
two snippets are from memory of Microsoft's "Accessing network applications with WSL" page —
check them there before running:

```ini
# %UserProfile%\.wslconfig
[wsl2]
networkingMode=mirrored
```

```powershell
# elevated PowerShell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
wsl --shutdown
```

Also allow UDP 4010 through Windows Defender Firewall. `potpori chat` waits up to 15 minutes
for a tail that is not reachable yet, and says so while it waits.

## When it does not work

| You see | Meaning |
|---|---|
| `the chain refused the session — tail [18, 36): …` | The tail checked the plan against how it was started and said no; the text after the node name is the reason (token, model, layers, `--chunk`, `--max-seq`, version). |
| `could not reach <addr> within 900s` | Nothing answered: wrong IP or port, a firewall, the tail is not up, or (layout B) WSL networking. |
| `bad frame: payload is N bytes, expected 2N` | The residual stream left the head in the wrong dtype. Should be impossible after the F32-boundary fix; if it appears, the sending backend is not honouring the in-graph cast. |
| `draft sanity check failed: first prediction is token 0` | The draft model's kernel produced NaN (a miscompiled kernel for this `max_seq` or dtype). |
| Fluent for a few tokens, then garbage; tok/traversal near 1.0 | The two halves disagree numerically — most likely NaN or overflow on one side. Note the backends and dtypes in the `[link: …]` line. |
| `cannot find function cuEventElapsedTime_v2` at build | A CUDA toolkit older than 12.8 is on `PATH` (`nvcc --version`). |

Two things have never run before this test, so treat them as the likely suspects: a **CUDA
shard exchanging a residual stream** (that boundary was fixed by reading the runtime's source,
not by running it — until now only the whole-model CUDA path had run), and **F32-on-Metal
talking to Bf16-on-CUDA** in one chain.

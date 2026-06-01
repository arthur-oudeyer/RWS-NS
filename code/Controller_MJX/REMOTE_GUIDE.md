# MJX Remote Training — Quick Reference

## Environment setup (one-time)

```bash
module load Mamba/23.11.0-0
mamba create -n mjx python=3.11 -y
mamba activate mjx
pip install -U "jax[cuda12]"
pip install mujoco flax optax "imageio[ffmpeg]" pillow
```

Add to `~/.bashrc` so every SSH session is ready:
```bash
module load Mamba/23.11.0-0
conda activate mjx
```

Verify:
```bash
python -c "import jax; print(jax.devices())"        # → [CudaDevice(id=0) ...]
python -c "from mujoco import mjx; print('MJX OK')"  # → MJX OK
```

---

## Per-session activation

```bash
module load Mamba/23.11.0-0   # skip if already in ~/.bashrc
conda activate mjx
cd ~/RWS-NS/code/Controller_MJX
```

---

## CLI commands

### Train from scratch (random reward weights)
```bash
CUDA_VISIBLE_DEVICES=0 python controller_cli_mjx.py new \
    --steps 2000000 --envs 1024 --rollout 128
```

### Warm-start from an existing policy (mutated reward)
```bash
CUDA_VISIBLE_DEVICES=0 python controller_cli_mjx.py mutate \
    cli_output/run_0001_new/policy.params \
    --steps 1000000 --envs 1024 --rollout 128
```

### Train with manually edited reward weights
```bash
# 1. Copy a reward file and edit it
cp cli_output/run_0001_new/reward.json my_weights.json
# edit my_weights.json with any text editor

# 2. Train with it
CUDA_VISIBLE_DEVICES=0 python controller_cli_mjx.py manual \
    my_weights.json --steps 2000000 --envs 1024 --rollout 128
```

### Re-render a saved policy (no training)
```bash
CUDA_VISIBLE_DEVICES=0 python controller_cli_mjx.py render \
    cli_output/run_0001_new/policy.params
```

### GPU benchmark
```bash
CUDA_VISIBLE_DEVICES=0 python controller_cli_mjx.py bench --full
```

---

## All flags

| Flag            | Default   | Description                                      |
|-----------------|-----------|--------------------------------------------------|
| `--steps N`     | 200 000   | Total PPO training steps                         |
| `--envs N`      | 512       | Parallel environments (vmap). See tuning below.  |
| `--rollout N`   | 64        | Steps per PPO rollout scan                       |
| `--seed N`      | random    | PRNG seed                                        |
| `--episode F`   | 5.0       | Episode duration in seconds                      |
| `--fall-height F` | 0.3     | Torso-z fall threshold                           |
| `--arch N [N…]` | 256 256   | Policy hidden layer sizes                        |
| `--sigma F`     | 0.8       | Reward init sigma (`new` only)                   |
| `--sigma F`     | 0.3       | Mutation sigma (`mutate` only)                   |
| `--out DIR`     | cli_output| Output root directory                            |

---

## Training speed tuning (RTX 2080 Ti)

### Why is the first run slow?

JAX compiles the entire rollout + PPO update into a single XLA kernel on the
**first update only** (~30–60 s). All subsequent updates run at full GPU speed.
The fps counter in the output is cumulative (total steps / total time), so it
starts low and climbs toward the true throughput as updates accumulate.

**With only 6 updates (200k steps / 512 envs / 64 rollout), most reported time
is compile time — not real training. The fps at update 6 is still not the
plateau. Use ≥ 50 updates to measure real throughput.**

### Rule of thumb: how many updates?

```
n_updates = steps / (envs × rollout)

200 000 / (512 × 64)  =   6  updates  ← mostly compile time
2 000 000 / (1024 × 128) = 15  updates  ← getting there
5 000 000 / (1024 × 128) = 38  updates  ← good measurement
```

### Recommended parameters for RTX 2080 Ti (11 GB VRAM)

After contact-buffer + arithmetic-intensity fixes, expected throughput on 2080Ti:

| envs   | rollout | GPU-Util | Steady-state |
|--------|---------|----------|--------------|
| 1024   | 128     | ~36 %    | ~7 000 steps/s    |
| 4096   | 128     | ~60 %    | ~20 000 steps/s   |
| 8192   | 128     | ~90 %    | ~38 000 steps/s   |
| 16384  | 64      | ~95 %    | ~50 000+ steps/s  |

| Goal                          | --steps    | --envs | --rollout | Updates |
|-------------------------------|-----------|--------|-----------|---------|
| Quick behavior check (~1 min) | 5 000 000 | 8192   | 128       |  5      |
| Meaningful training (~3 min)  | 25 000 000| 8192   | 128       | 24      |
| Solid training (~10 min)      | 75 000 000| 8192   | 128       | 71      |
| Saturating (~30 min)          | 250 000 000| 8192  | 128       | 238     |

### Why does `nvidia-smi` always show ~8500 MiB?

That's **JAX's preallocator**, not actual usage. JAX grabs 92% of VRAM at startup
(configurable via `XLA_PYTHON_CLIENT_MEM_FRACTION`) and manages allocations
internally. To see real usage:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python controller_cli_mjx.py new ...
```

Without preallocation you'll see actual usage scale with `--envs`. Use it for
debugging only — preallocation is faster at runtime.

### Tuning `--envs`

Bigger is better up to GPU-Util saturation, then up to VRAM cap.
For this morphology with rollout=128:

```bash
#   --envs  1024  → GPU-Util  ~36 %  (way under-utilized)
#   --envs  4096  → GPU-Util  ~60 %
#   --envs  8192  → GPU-Util  ~90 %  (recommended sweet spot)
#   --envs 16384  → may OOM with rollout=128; try rollout=64
```

### Tuning `--rollout`

Longer rollout = better arithmetic intensity per JIT iteration BUT eats more
VRAM (rollout buffer scales linearly).  With `envs=8192`:
- `rollout=64`  → ~256 MB buffer, OK to push envs higher
- `rollout=128` → ~512 MB buffer (recommended)
- `rollout=256` → ~1 GB buffer, fewer envs

### Persistent JIT cache

The CLI now caches the XLA-compiled training kernel to `~/.cache/jax_mjx/`.
First run compiles (~60 s), subsequent runs with the **same** `--envs/--rollout/
--arch` skip compile entirely. Different params → different cache entry.

If you ever want to clear it (e.g. after a JAX version upgrade):
```bash
rm -rf ~/.cache/jax_mjx
```

### Warm-start is cheaper than from-scratch

Within one CLI invocation, `mutate` reuses the in-memory JIT kernel. Across
invocations, the persistent cache makes them equally fast after the first run.

---

## Output layout

```
cli_output/
  run_0001_new/
    policy.params    Flax weights (pickle)
    video.mp4        2-camera rollout (side-by-side)
    reward.json      Reward weights used for this run
    info.json        Fitness, steps/s, device, seed, …
  run_0002_mutate/
    …
  log.jsonl          One JSON line per completed run
```

## Copying results back to local machine

```bash
# From your local machine (single-hop SSH):
scp -r remote_host:~/RWS-NS/code/Controller_MJX/cli_output/run_0001_new/ .

# With a jump host:
scp -J jump_host remote_host:~/RWS-NS/code/Controller_MJX/cli_output/run_0001_new/ .

# Whole output folder as a tarball (faster over slow links):
# On remote:
tar czf results.tar.gz cli_output/
# On local:
scp -J jump_host remote_host:~/RWS-NS/code/Controller_MJX/results.tar.gz .
tar xzf results.tar.gz
```
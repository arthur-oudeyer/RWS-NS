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

| Goal                          | --steps    | --envs | --rollout | Updates |
|-------------------------------|-----------|--------|-----------|---------|
| Quick behavior check (~2 min) | 1 000 000 | 512    | 128       | 15      |
| Meaningful training (~5 min)  | 3 000 000 | 1024   | 128       | 23      |
| Solid training (~15 min)      | 10 000 000| 1024   | 256       | 38      |
| Full run (~1 hour)            | 50 000 000| 2048   | 256       | 95      |

### Tuning `--envs`

More envs = better GPU utilisation up to a point; too many = VRAM overflow.

```bash
# Find the sweet spot: increase until VRAM is ~80% full or fps stops rising
nvidia-smi  # check MiB used after first update

# Rough guide for 2080Ti (11 GB):
#   --envs 512   → safe baseline
#   --envs 1024  → likely sweet spot
#   --envs 2048  → may hit VRAM limit depending on model size
```

### Tuning `--rollout`

Longer rollout = fewer JIT entries = higher effective GPU throughput.
Tradeoff: longer rollout = noisier gradient estimates (usually acceptable).
Recommended: 128 or 256.

### Warm-start is cheaper than from-scratch

`mutate` reuses JIT-compiled kernels from the same session if run back-to-back.
The first `new` in a session pays the compile cost; subsequent `mutate` calls
on the same GPU are much cheaper.

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
# 08 – Multi-Host TPU Pod Training

Scales training across multiple TPU hosts using `jax.distributed`. This is the only example in the set that crosses a host boundary, and these patterns are what make pod-scale training (and pod-scale inference) work.

## Concepts

- `jax.distributed.initialize(coordinator_address, num_processes, process_id)` — connects every host to one coordinator (process 0)
- `jax.local_devices()` — cores attached to **this** host only
- `jax.devices()` — **all** cores in the entire pod (all hosts)
- `pmap` + `jax.lax.pmean(axis_name="batch")` — gradient all-reduce across the full pod
- Each host generates its own shard of the global batch (RNG seeded by `process_index`)

## Topology

```
Pod (e.g. v3-32 = 4 hosts × 8 cores = 32 chips)
├── Host 0  (coordinator)  ← run with --process_id=0
├── Host 1                 ← run with --process_id=1
├── Host 2                 ← run with --process_id=2
└── Host 3                 ← run with --process_id=3
```

Each host has its own CPU, RAM, and 8 attached TPU cores. Device↔device communication uses the TPU ICI (inter-chip interconnect). Host↔coordinator control traffic uses the VPC.

## Hardware context

Multi-host matters once a single host's HBM is insufficient. For inference, single-chip VMs are sufficient for all 75 benchmark models:

| Pod | Chips | Hosts | v5e aggregate HBM | v6e aggregate HBM | Use case |
|-----|-------|-------|-------------------|-------------------|----------|
| v5e-1 / v6e-1 | 1 | 1 | 16 GB / 32 GB | — | Benchmark harness (all 75 models) |
| v5e-8 / v6e-8 | 8 | 1 | 128 / 256 GB | — | Large batch inference; single-host pmap |
| v3-32 / v5e-32 | 32 | 4 | 512 GB | 1 TB | This example; frontier-model training |
| v5e-256 | 256 | 32 | 4 TB | — | LLM pretraining at scale |

Even GPT-2-XL (1.5 B params × 4 bytes = 6 GB weights) fits in a single v5e-1's 16 GB with KV-cache headroom. Multi-host is reserved for cases the benchmark explicitly flags as pod-only (none in Stage 1–8).

## Run (one command per host)

```bash
# Host 0 (coordinator)
python train.py \
  --coordinator_address=10.0.0.1:8476 \
  --num_processes=4 \
  --process_id=0

# Host 1
python train.py \
  --coordinator_address=10.0.0.1:8476 \
  --num_processes=4 \
  --process_id=1

# Host 2
python train.py \
  --coordinator_address=10.0.0.1:8476 \
  --num_processes=4 \
  --process_id=2

# Host 3
python train.py \
  --coordinator_address=10.0.0.1:8476 \
  --num_processes=4 \
  --process_id=3
```

All four commands must be running for `jax.distributed.initialize` to return. The coordinator (host 0) blocks until every process has registered.

## Single-host smoke test

```bash
python train.py   # omit --coordinator_address → single-process mode
```

Runs as a degenerate single-host case — `pmean` becomes a local all-reduce only. Confirms the script is correct before booking a pod.

## Expected output (host 0 only, 200 steps)

```
[process 0]  local=8 device(s)  global=32 device(s)
Starting training on 32 total device(s) ...
step  50  loss=2.143  acc=18.4%
step 100  loss=1.872  acc=25.1%
step 150  loss=1.641  acc=31.7%
step 200  loss=1.453  acc=38.2%
```

Hosts 1–3 print nothing — the script gates printing on `jax.process_index() == 0`.

## What to observe

- **`jax.devices()` returns 32, `jax.local_devices()` returns 8.** This is the definitive confirmation the pod is wired up. If `jax.devices()` shows 8, you're single-host and the cross-host all-reduce isn't happening.
- **First-compile is per-host but concurrent.** Each host independently compiles the same XLA program — on a fresh pod that's 4 × 25 s in parallel, so still ~25 s wall time.
- **`pmean` is global.** Even though `pmap` is over local devices, the `axis_name="batch"` reduction crosses the ICI to all 32 chips. Identical loss values across all hosts confirm this.
- **RNG sharding by `process_index`.** Each host must generate a *different* shard of the global batch. If you seed all hosts identically, all 4 process the same data → wasted compute, wrong gradient.
- **Failure modes.** If any host dies mid-training, the next collective hangs. Orbax checkpoints (example 03) are how you recover — save every N steps, not just every epoch.

## Connection to the benchmark

Stage 1–8 of the benchmark **does not** use multi-host. Every model in the 75-model registry fits on a single chip for inference, and simpler environments produce more reproducible measurements (`observe/stats.py`: n=3, Grubbs, CV<10%).

This example exists for two forward-looking reasons:

1. **Frontier models (Stage 9+):** When the registry expands to 70B+ LLMs, the same `jax.distributed` pattern is the entry point. The `local_devices` vs `devices()` distinction is the only API change from single-host `pmap`.
2. **Pod-scale serving:** Multi-host inference (one request fanned across hosts, model-parallel) reuses the coordinator + `local_devices` pattern. Path 1 (JAX+TPU) supports this cleanly; Path 5 (HF Inference API) hides it; Paths 2/3 would need NCCL multi-node, which is a different story.

For now, treat this as the reference for "how host boundaries work in JAX" — so you're not learning it under pressure when the benchmark expands.

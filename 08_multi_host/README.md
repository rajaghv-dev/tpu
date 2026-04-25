# 08 – Multi-Host TPU Pod Training

Scales training across multiple TPU hosts (a TPU pod slice) using `jax.distributed`.

## Concepts
- `jax.distributed.initialize` — connects all hosts to a single coordinator
- `jax.local_devices()` vs `jax.devices()` — local cores vs the full pod
- `pmap` + `jax.lax.pmean` — gradients are averaged across **all** cores on **all** hosts
- Each host generates its own shard of the global batch

## Topology
```
Pod (e.g. v3-32 = 4 hosts × 8 cores)
├── Host 0  (coordinator)  ← run with --process_id=0
├── Host 1                 ← run with --process_id=1
├── Host 2                 ← run with --process_id=2
└── Host 3                 ← run with --process_id=3
```

## Run (one command per host)
```bash
# Host 0
python train.py \
  --coordinator_address=10.0.0.1:8476 \
  --num_processes=4 \
  --process_id=0

# Host 1
python train.py \
  --coordinator_address=10.0.0.1:8476 \
  --num_processes=4 \
  --process_id=1
```

## Single-host smoke test
```bash
python train.py   # no --coordinator_address → single-process mode
```

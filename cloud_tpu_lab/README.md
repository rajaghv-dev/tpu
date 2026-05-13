# cloud_tpu_lab

Hands-on Google Cloud TPU learning + architecture analysis.

**Goal:** learn the full Cloud TPU stack — hardware, XLA, PJRT, sharding,
profiling, cost — by running executable code, not just reading docs.

**Three run modes:**

1. **Local / CPU simulation** — no TPU, no GCP, no sudo. The default.
   Runs on stdlib only.
2. **Google Colab** — CPU fallback by default; auto-detects TPU runtime.
3. **Cloud TPU VM** — optional scripts in `gcp/` to provision a real TPU
   VM, run benchmarks, collect artefacts, then **clean up** to stop billing.

## 60-second start (no TPU required)

```bash
cd cloud_tpu_lab
python3 examples/run_cpu_simulation_demo.py
```

That runs the end-to-end vertical slice:

> tiny model → fake HLO → fake XLA compile → fake PJRT runtime
> → fake TPU device exec → HBM sim → sharding sim → profiler trace
> → JSONL log + CSV metrics → cost estimate → bottleneck report

Output lands in `cloud_tpu_lab/artifacts/`:

- `logs/run_<trace_id>.jsonl` — every event with correlation IDs
- `metrics/run_<trace_id>.csv` — Prometheus-style metric stream
- `traces/run_<trace_id>.json` — Chrome-trace JSON
- `reports/run_<trace_id>.md` — the report to open first

Compare TPU versions side by side:

```bash
python3 examples/run_cpu_simulation_demo.py --show-versions
```

## Try different knobs

```bash
# Different TPU spec → different HBM, different step time, different cost
python3 examples/run_cpu_simulation_demo.py --tpu-version v5p --chip-count 4
python3 examples/run_cpu_simulation_demo.py --tpu-version v6e --batch-size 64

# Heavier model → HBM pressure + longer compile
python3 examples/run_cpu_simulation_demo.py --hidden-size 1024 --num-layers 12
```

## Repo layout

```
cloud_tpu_lab/
├── docs/         Learning material (00–15 modules)
├── notebooks/    Run-along Jupyter notebooks
├── src/          Simulation + observability code
│   ├── common/         trace IDs, configs, cost
│   ├── tpu_versions/   Catalog (v4 · v5e · v5p · v6e)
│   ├── xla_sim/        Fake HLO + lowering + compile cache
│   ├── pjrt_sim/       Fake PJRT runtime + executable + device
│   ├── memory/         HBM + activation + checkpoint estimators
│   ├── sharding/       Mesh + partitioner + collectives
│   ├── input_pipeline/ Dataloader / prefetch sim
│   ├── profiling/      Profiler trace + analyzer + bottleneck report
│   ├── observability/  JSONL logger + CSV metrics + Markdown report
│   ├── traceability/   Join-by-trace_id across artefacts
│   └── model_examples/ Tiny MLP / transformer in JAX / Torch-XLA / TF
├── gcp/          Cloud TPU VM provisioning + cleanup scripts
├── examples/     Runnable demos
├── tests/        Minimal smoke tests (no real TPU needed)
├── artifacts/    Generated logs / metrics / traces / reports / plots
└── observability/
    ├── docker-compose.yml  (Prometheus + Grafana + Loki + Tempo)
    ├── prometheus/
    ├── grafana/
    ├── loki/
    ├── tempo/
    └── exporters/          (prometheus_client exporter)
```

## OCT model — Observability / Controllability / Traceability

Every event in the simulation carries the same correlation ID bundle:

```
trace_id
├── step_id
├── model_layer_id
│   └── hlo_op_id
│       └── executable_id
│           └── device_event_id
│               ├── tensor_id → shard_id
│               └── collective_id
```

This is the spine of the OCT model. See `docs/13_oct_metrics_dictionary.md`.

## Cost safety

- Every `gcp/*.sh` script that creates a paid resource has a matching
  cleanup script in the same directory.
- Pricing is never hardcoded. Pass `--hourly-usd-per-chip` and look up the
  current rate at https://cloud.google.com/tpu/pricing.
- Idle TPU VMs accrue cost even when idle — **always run `delete_tpu_vm.sh`
  when done.**

## Tests

```bash
make smoke        # ~1 second — what CI runs
make test         # full suite, still no TPU needed
```

## License

MIT — see `LICENSE` (or the repo metadata).

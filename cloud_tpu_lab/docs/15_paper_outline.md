# 15 — Paper / Blog Post Outline (scratch)

> **Learning goal:** sketch a publishable narrative for the work in
> `cloud_tpu_lab` — not a finished paper, just the skeleton. Each section
> is 4–8 bullets. The goal is to have a structure to drop findings,
> charts, and numbers into as the lab matures.

This document is intentionally a **scratch outline**. It is not the
artefact you would publish; it is the spine to fill in. When a section
becomes solid, port it to a stand-alone document under `docs/` or to a
draft post.

Conventions used in the bullets below:

- "Number" means "a specific measured value to drop in here later".
- "Chart" means "a screenshot or extracted plot from
  `artifacts/.../reports/`".
- "Sim" means "data from `examples/run_cpu_simulation_demo.py`".
- "Real" means "data from a Cloud TPU VM run that has since been deleted
  per `11_cleanup_and_cost_safety.md`".

---

## 1. Working title

- Working title: "Observable, Controllable, Traceable: A Field Guide to
  Cloud TPU Performance Engineering".
- Subtitle option A: "From simulator to silicon without surprises".
- Subtitle option B: "Three artefacts, one trace_id, every layer of the
  stack".
- Audience: ML engineers who have access to Cloud TPU and want a
  framework, not a leaderboard.
- One-line pitch: a vertical-slice methodology — simulator + real TPU +
  observability stack — so you can predict, then verify, then explain a
  TPU run.

---

## 2. Abstract (4–6 bullets)

- One paragraph stating the problem: TPU performance work is mostly tribal
  knowledge; numbers don't reproduce across teams.
- One paragraph stating the contribution: the OCT framework
  (Observability / Controllability / Traceability) operationalised in a
  small open lab.
- A claim about what the simulator predicts within X% of the real TPU
  (Number to be filled).
- A claim about cost-per-token across N SKUs (Number, Chart).
- A claim about how recompile-loop detection saved Y hours of wall-time
  (Number).
- A pointer to the lab as runnable artefact.

---

## 3. Background (6–8 bullets)

- Brief history of TPU generations relevant to the work (v4 → v5e → v5p →
  v6e), as enumerated in `src/tpu_versions/`. Do not editorialise on
  pricing — point at https://cloud.google.com/tpu/pricing.
- Frameworks landscape: JAX, PyTorch-XLA, TensorFlow on TPU. Why this
  paper treats them as equivalent producers of OCT signals.
- XLA / HLO / PJRT in one paragraph each. Cross-reference
  `src/xla_sim/` and `src/pjrt_sim/`.
- Sharding lexicon: data-parallel, model-parallel, fully-sharded,
  tensor-parallel. The lab models all four via meshes in
  `src/sharding/`.
- HBM and the chip-bandwidth wall. Cross-reference `src/memory/`.
- The cost-and-cleanup problem nobody writes about. Cross-reference
  `11_cleanup_and_cost_safety.md`.
- Prior art: a one-paragraph summary of public TPU benchmarking work
  (no fabrications — fill in as you find references).

---

## 4. The OCT framework (4–6 bullets)

- Definition of the three letters and what each one means in the context
  of a TPU run.
- The correlation-ID spine: `trace_id → step_id → model_layer_id →
  hlo_op_id → executable_id → device_event_id → tensor_id / shard_id /
  collective_id`. Source: `src/common/trace.py`.
- The three artefact streams: metrics CSV (Prometheus), JSONL logs
  (Loki), span traces (Tempo). One join key, three stores.
- Cardinality discipline as a first-class design constraint. Reference
  the `SAFE_LABELS` / `DANGEROUS_LABELS` split in
  `src/observability/metrics.py`.
- The metric vocabulary: canonical names enumerated in
  `13_oct_metrics_dictionary.md`. Argue for stable names across
  simulator and silicon.
- The bottleneck report as a deterministic mapping from signal pattern
  to recommended fix. Source: `src/profiling/bottleneck_report.py`.

---

## 5. The simulator (5–8 bullets)

- Why a simulator: rehearse the methodology without paying.
- Architecture: tiny model → fake HLO → fake XLA compile → fake PJRT
  runtime → fake device execution → HBM sim → sharding sim → profiler →
  observability. Cross-reference the layout in `src/`.
- What is faithfully modelled: step-time breakdown, compile cost
  asymmetry, HBM capacity / OOM, collective scaling pattern.
- What is *not* modelled: real microarchitectural effects, dynamic
  hardware sharing, kernel autotuning. Be honest about this.
- The TPU-version catalog in `src/tpu_versions/`: per-SKU peak FLOPs,
  HBM size, chip topology. Pricing is **not** in the catalog.
- Side-by-side comparison example:
  `python3 examples/run_cpu_simulation_demo.py --show-versions`.
- A worked simulator run, end-to-end, with output snippets from the
  generated `artifacts/reports/run_<trace_id>.md`. Chart.
- Limitations: a sentence each, pointing at the bullets of section 9.

---

## 6. Real Cloud TPU validation (5–8 bullets)

- Methodology: follow `10_cloud_tpu_setup_playbook.md` end-to-end on N
  SKUs (start with `v5litepod-1` and `v6e-1`). Cleanup per
  `11_cleanup_and_cost_safety.md`.
- Workloads: a tiny matmul, a small MLP, a small transformer. All in
  `src/model_examples/`. Same code, three frameworks.
- For each (workload × SKU × framework): cold compile run + warm
  compile run + steady-state 30-step window + 3-process repeat.
- Reporting protocol: median, p95, p99, HBM util, MXU util, MFU,
  cost-per-step, cost-per-token. As per section 7 of
  `14_benchmarking_playbook.md`.
- Cross-reference the bottleneck-report findings produced for each run.
- Predicted vs measured: a single chart per metric with the simulator's
  prediction overlaid. Number: typical % delta.
- Where the prediction was wrong, and why. Be specific.
- One real-life "recompile loop caught early" anecdote — Number for
  wall-time saved.

---

## 7. Cost / performance methodology (5–7 bullets)

- Why cost-per-token, not cost-per-step or cost-per-run, is the right
  KPI.
- The lineage protocol: `config.json`, `env.json`, `git.json`,
  `pricing.json`, `hardware.json`. Source:
  `14_benchmarking_playbook.md` section 8.
- The pricing discipline: never hardcode. Always look up at
  https://cloud.google.com/tpu/pricing and persist the value with the
  run. Date-stamp it.
- The 3-run minimum and the median-of-medians.
- Cleanup as part of methodology: a benchmark whose VM was deleted late
  has a cost number that is wrong. Reference
  `11_cleanup_and_cost_safety.md`.
- Reproducibility test: re-run an old result from lineage only. Number:
  delta vs original.
- A short table: cost-per-token across the N SKUs studied (Numbers).

---

## 8. Observability stack in practice (4–6 bullets)

- The two modes: no-install JSONL/CSV/MD vs local Grafana stack. From
  `12_observability_with_grafana_prometheus.md`.
- A walk-through: spike in `cloud_tpu_step_time_seconds` → Loki pivot
  on `trace_id` → Tempo span tree → recommended fix.
- Cardinality lessons: which Prometheus labels we tried that blew up,
  and what we did instead. (Sim screenshot of head-series growth.)
- Dashboards shipped with the lab and what each is for. Chart.
- Alerts: thresholds chosen to match
  `src/profiling/bottleneck_report.py` rules.
- Honest note on cost of running the local stack itself — it's free
  (Docker on a laptop), but operating it daily is mental overhead.

---

## 9. Limitations (4–6 bullets)

- Simulator does not model microarchitectural effects (NUMA on host,
  shared TPU pod neighbours, autotuning).
- Real-TPU runs in the paper are at small chip counts. Big-pod
  behaviour is out of scope; the methodology should still apply but
  has not been validated by us.
- Cost numbers are time-stamped and SKU-specific; they will date.
- The bottleneck report rules are thresholds chosen by judgement, not
  by formal optimisation. Document them and let readers tune.
- No claim about parity with vendor profilers (XProf); the OCT
  framework is complementary, not a replacement.
- The OCT label / span attribute taxonomy may need extension as new
  workloads (e.g. MoE, inference-serving) bring new layers.

---

## 10. Future work (4–6 bullets)

- Extend the simulator to model pod-network topology effects more
  faithfully (cross-rack collectives, host-mesh overhead).
- Add an inference-serving workload class (different observability
  needs, different cost KPI).
- Auto-derive bottleneck thresholds from a corpus of recorded runs,
  rather than hand-tuning them.
- Integrate with vendor profilers: parse XProf output into the same
  OCT artefact shape so both can be analysed by `src/profiling/`.
- Multi-region / spot / reservation cost modelling — currently the lab
  treats pricing as a scalar.
- A "lab in CI" pattern: spin up a tiny TPU VM nightly, run a known
  workload, post the OCT artefacts as a PR comment, delete the VM.

---

## 11. Conclusion (3–4 bullets)

- Restate the framework in one sentence.
- Restate the headline measurement from section 6 with the Number(s)
  that supported it.
- One sentence on the cost-safety contribution.
- One sentence on what the reader should do next (run the simulator,
  then book a TPU for 30 minutes, then delete it).

---

## 12. Appendix candidates

- **A.** Metric dictionary verbatim from `13_oct_metrics_dictionary.md`.
- **B.** JSONL schema verbatim.
- **C.** Span name table.
- **D.** Cleanup checklist verbatim from
  `11_cleanup_and_cost_safety.md`.
- **E.** Pricing lookup procedure + how to date-stamp it.
- **F.** Reproducibility test protocol.
- **G.** Sample artefact tree (one full run).

---

## 13. Cross-references inside the lab

- `README.md` — 60-second start.
- `10_cloud_tpu_setup_playbook.md` — setup.
- `11_cleanup_and_cost_safety.md` — cleanup.
- `12_observability_with_grafana_prometheus.md` — stack.
- `13_oct_metrics_dictionary.md` — vocabulary.
- `14_benchmarking_playbook.md` — methodology.
- `src/observability/` — producers.
- `src/profiling/` — consumers.
- `src/tpu_versions/` — SKU catalog.

---

## 14. Exercises / TODOs

1. Pick one bullet under section 2 (Abstract). Replace its "Number"
   placeholders with real values from a recent simulator run. Note what
   would change if the same exercise used real-TPU data.
2. Draft a single figure for section 6 (Real Cloud TPU validation) that
   plots simulator-predicted vs measured `cloud_tpu_step_time_seconds`
   for one workload across two SKUs.
3. Identify a section that is *not yet* fillable because the lab hasn't
   produced the data. Add the gap to `progress_log.md` under the next
   suggested milestone.
4. Write a one-paragraph "what surprised us" insert for section 9
   (Limitations). Honesty here is the most valuable part of the paper.
5. When section 7 (Cost / performance methodology) has at least 3
   measured rows, port it out of this outline into a stand-alone
   `docs/16_cost_perf_results.md`.

# Prompts & Requirements Log

Running record of user prompts and intent for this repo.
Improved versions are the primary reference. Original raw prompts are preserved at the bottom.

---

## Improved Prompts (Opus-rewritten, 2026-04-26)

**P1 — Initialize TPU examples repo**
Create a new Git repository scaffold for running multiple inference examples on Google Cloud TPU. Include: directory layout (`examples/`, `src/`, `scripts/`, `configs/`, `docs/`), a `README.md` with setup steps, `requirements.txt` pinned to JAX/Flax + TPU-compatible versions, a `.gitignore` for Python/JAX artifacts, and one minimal runnable JAX-on-TPU example (e.g., matmul or a small forward pass) as a sanity check. Output the full file tree plus the contents of each file.

**P2 — Add gcloud tooling**
Extend the existing TPU repo with gcloud CLI integration. Add: a `scripts/gcloud/` directory containing shell scripts to (a) authenticate, (b) create/delete a TPU VM (v5e-1 and v6e-1 variants), (c) SSH into the VM, (d) sync code via `gcloud compute scp`, and (e) tear down resources to avoid idle billing. Include a `docs/gcloud_setup.md` with prerequisites, IAM roles needed, and a step-by-step first-run walkthrough. Output the scripts and docs only — no changes to existing example code.

**P3 — Configure personal Google account**
Update the gcloud scripts and docs to use my personal Google account `rajaghv@gmail.com` (alternate `rajaghv.dev@gmail.com`) on a personal GCP project. Replace any placeholder account/project values, document how to set the active account (`gcloud config set account`), and note how to switch between the two emails. Keep the scripts idempotent.

**P4 — Observability design ideas**
I want to compare TPU vs GPU inference on the same models with full observability. Without writing code yet, propose a design for an observability stack that captures: device-level metrics (utilization, memory, power, thermals where available), per-step latency/throughput, compile time vs run time, host-side overhead, and trace-level profiling. List candidate tools per side (JAX/TF profiler, XLA dumps, Nsight, PyTorch Profiler, DCGM, etc.), what dimensions are directly comparable vs proxy-comparable, and a recommended unified schema for storing results. Output as a structured doc with sections: Goals, Metrics, Tools per Backend, Comparability Matrix, Storage Schema, Open Questions.

**P5 — Extend repo to GPU + sync, no code yet**
Add TPU profiler integration (account: `rajaghv@gmail.com`) and extend the repo to run the same experiments on local GPUs so results can be compared across multiple dimensions. Results should sync back to GitHub; experiments should run locally on my GPU machines. Do NOT write code yet — only discuss the design. Also: maintain a `prompts.md` file in the repo and append every prompt I send going forward (verbatim). Output: a design discussion covering repo restructure, GPU runner plan, results sync strategy, and confirmation that `prompts.md` has been created/updated.

**P6 — Cost analysis for personal account**
My GPUs: RTX 3080 16GB, RTX 4090 24GB, DGX Dell box (256GB system RAM). I'll follow your recommendations. Focus: comparing single-card TPUs vs single-card GPUs across multiple available variants. Using my personal GCP account, estimate the cost to me for running benchmarks on TPU (v5e-1, v6e-1, and any other accessible single-card variants), assuming preemptible where possible. I also have Colab Pro — factor in what it can substitute. Output: a cost table (per-hour, per-experiment-run estimate, monthly cap suggestion), a recommended TPU/GPU shortlist, and a brief rationale. No code.

**P7 — Colab Pro CLI + deep short experiments**
Can Colab Pro be driven from a CLI (e.g., headless/automated) for our experiments? Goal: each deep experiment finishes in under 20 minutes and produces results comparable across these axes — compiler (XLA, TorchDynamo, etc.), ecosystem (JAX, PyTorch), model variants, pruning, quantization, microarchitecture behavior, and dataflow — across vision, NLP, audio, and multimodal models. Output: (1) feasibility and mechanics of CLI-driving Colab Pro, (2) a proposed experiment template that fits the <20-minute budget, (3) the full set of comparison dimensions structured as a matrix, (4) recommended model coverage per modality. No code yet.

**P8 — Single-card, sequential, preemptible strategy**
Revise the strategy with these constraints: stack is JAX + PyTorch (I'll defer to your recommendations on specifics); experiments run sequentially, not in parallel; each run is 1–3 minutes with a small fixed input set per experiment; single-card only; preemptible cloud VMs are acceptable for my personal cost. Re-output: experiment design, scheduling/queue approach, expected per-experiment cost, and any changes to the previously proposed observability and repo plan. No code.

**P9 — Inference-only scope lock**
Lock the scope of this repo to inference only. Remove or defer any training/fine-tuning plans from the strategy, model list, observability design, and cost estimates. Re-confirm the updated scope in one short section.

**P10 — Add torch_xla, HF, model size cap**
Add torch_xla as a fourth execution path. I have a HuggingFace account — integrate it into the repo (auth, model pulls, optional inference endpoints). Optimize the design to keep per-experiment cost minimum. Expand the model list to include the latest open-source models up to ~4B parameters, and tell me what the maximum model size is that fits on each Google TPU single-card variant accessible to personal-account VMs (v5e-1, v6e-1, others). Output: updated execution-path matrix (JAX+TPU, JAX+GPU, PyTorch+GPU, torch_xla+TPU, HF), TPU-card-vs-max-model-size table with justification, and the expanded model list. No code.

**P11 — Persist context, expand model list strategically**
Append all of the above design decisions to `context.md` and all my prompts so far to `prompts.md`. Then, using your strongest reasoning, propose an additional curated list of models worth benchmarking on GPU vs TPU, chosen to maximize learning across architecture (attention variants, MoE, SSM, conv, hybrid), runtime (eager/graph/compiled), compile pipelines (XLA/Inductor/TensorRT), and model families/variants. For each model, give a one-line justification of what it teaches. Output: confirmation of `context.md` and `prompts.md` updates, plus the new model list as a table (Model | Family | Why it matters | Modality | Size).

**P12 — GCS + HuggingFace inference options**
GCS is acceptable for results storage. I have a HuggingFace account — can I use HF Inference (Inference API / Inference Endpoints / serverless) as an additional execution path in this benchmark, and how does it fit alongside the local/cloud paths? Also suggest any other low-cost or pedagogically useful execution paths I'm missing. Output: pros/cons of each HF inference mode for our benchmark, recommended integration approach, and any new paths to consider with rationale.

**P13 — Improve README readability**
Rewrite the GitHub `README.md` so it is clean and easy to read at a glance. Include: a one-paragraph project pitch, goals, scope (inference-only), execution paths, hardware matrix, model coverage summary, repo layout, quickstart, and a link map to deeper docs (`context.md`, `prompts.md`, etc.). Use clear headings, tables where appropriate, and keep prose tight.

**P14 — Add popular OSS models, staged plan, logging, viz**
Expand the model list with Qwen, DeepSeek, Gemma, Phi, and other popular open-source models, selected by popularity and evaluation-leaderboard standing. Plan the repo build-out in stages rather than all at once, where each stage is informed by results from the previous stage's runs. Specify: (a) how runs are logged (what fields, where stored, how indexed), (b) what visualization and comprehension tooling we'll add (charts, dashboards, comparison views). Output: staged roadmap (Stage 1..N with deliverables and exit criteria), expanded model list with selection rationale, logging schema, and visualization plan. No code.

**P15 — Update docs + gap analysis for full traceability**
Update `context.md`, `prompts.md`, and the GitHub repo with the latest decisions. Then perform a gap analysis: what's missing for full observability and full traceability such that every claim from these experiments is backed by recorded evidence (raw metrics, configs, code SHA, hardware fingerprint, environment, seeds, etc.)? Output: list of gaps, suggested fixes, and a proposed evidence-chain design (what artifact backs which claim type).

**P16 — DGX Blackwell, gap fixes, model justifications, India access, H100/B200 cost**
Correction: my DGX is a Dell NVIDIA box, Blackwell architecture, 256GB system RAM (note: not the GPU VRAM). Fix all gaps in `context.md` accordingly and update it. Then revisit the full model list and, for each model, justify what it lets us understand or benchmark — with concrete facts (architecture detail, known optimizations, quantization friendliness, etc.) so I can learn from each entry. Also: I'm in India — confirm which GPUs and TPUs are actually accessible to me on personal accounts. Provide a cost comparison for running our benchmarks on H100 vs B200 cloud instances alongside my local hardware. Output: updated `context.md`, model-with-justification table, India-access matrix, H100 vs B200 cost table.

**P17 — Lesson plan beginner to expert**
Create a structured lesson plan that takes me from beginner to expert on every terminology and concept relevant to this repo: hardware (TPU/GPU microarchitecture, memory hierarchy, interconnects), compilers (XLA, Inductor, TensorRT, MLIR), runtimes, inference, quantization, pruning, profiling, and the JAX/PyTorch/torch_xla/HF ecosystems. Output as a tiered plan: Beginner → Intermediate → Advanced → Expert, with each tier listing topics, key terms to master, and what I should be able to do/explain at the end of that tier. Markdown, no code.

**P18 — Single repo, multi-artifact knowledge base**
Make sure this single repo serves as a complete vehicle for understanding hardware, compilers, and model inference/quantization in depth across multiple dimensions. List the multiple artifacts the repo should produce (e.g., benchmark reports, traces, plots, model cards, lesson notes, decision logs, evidence bundles), what each artifact contains, and where it lives in the repo. Output: artifact catalog as a table (Artifact | Purpose | Contents | Path | Generation trigger).

**P19 — Lesson plan adds: Colab Pro + HF paid, no code**
Extend the existing lesson plan so I can leverage my Colab Pro and my paid HuggingFace account effectively within this repo's scope. Cover essentials → expert. Stay strictly within the repo's context (TPU/GPU inference benchmarking, observability, traceability). Do NOT write any code. Output the full updated lesson plan as a single markdown file.

**P20 — Session + memory files for continuity**
Create `SESSION.md` and `MEMORY.md` in the repo so a future session can resume work without re-deriving context. `SESSION.md` should capture current state, decisions locked, next actions, and open questions. `MEMORY.md` should capture durable facts (hardware, accounts, scope, model list pointer, paths). Define a short protocol at the top of each file for how a new session should read them. Output both files in full.

**P21 — Reason silently, push, stop**
Do your full logical reasoning internally without consuming tokens in the next session. Then commit and push `SESSION.md` and `MEMORY.md` (plus any pending updates to `context.md`, `prompts.md`, `README.md`) to GitHub. After the push succeeds, stop — no further actions, no summary beyond the commit/push confirmation.

---

## Original Raw Prompts (verbatim, 2026-04-25)

**P1:** make a repo for multiple examples in Google TPU

**P2:** add gcloud as well to this repo

**P3:** use my pvt acc google rajaghv (rajaghv@gmail.com / rajaghv.dev@gmail.com)

**P4:** how do i add observability stack so that i can compare n contrast the same between TPU and GPU? give me ideas

**P5:** ok add tpu profiler, my google account is rajaghv@gmail.com. can u extend this repo for running same experiments on GPU so that i can compare n contrast in multiple perspectives. the results can be sync backed to github, and locally run on my gpu machines, dont code now, lets discuss and u keep my prompts in prompts.md file for now

**P6:** rtx3080 16gb, 4090 24gb, dgx with 256gb boxes, i go by your recommendations, focus is on multiple available tpus vs gpus, focusing on single card, what would be cost involved for me if i use personal account for tpu? i have google colab as well.

**P7:** i have google colab pro, is it possible to execute in cli for colab pro? the idea is to run deep examples in less than 20 mins and compare the results, from various aspects of compiler, ecosystem, model variants, pruning/quantization variants etc, in as much depth as possible to understand the microarch, their dataflow study based on various models for vision, nlp, audio, multimodal and all more possible.

**P8:** jax, pytorch, i consider ur approach, not planning to run multiple experiments in parallel, each run can take 1-3 mins with limited no.of inputs to the model per experiment, so redo the strategy again, im looking for single card experiments now, cloud preemptible vm is ok for my personal cost.

**P9:** thats inference only focus now.

**P10:** add torch xla, i have hugging face account, you can add it to this repo, ok whatever you suggest to keep the cost min per experiment, add more latest models till 4B or suggest whats the max per google tpu card, and its variants accessible to vms

**P11:** good thanks for all your inputs, add all the above to our context.md and prompts to prompts.md; identify another list of models that can be benchmarked in GPU vs TPU, more in-depth arch, runtime, compile, models etc variants. use your best model to strategize this brilliantly, effectively so that a lot more can be learnt by me from the multiple perspectives.

**P12:** gcs is accepted, i have hugging face account, can i use hf inferencing as well? suggest me more

**P13:** update my github so that i can read it legibly

**P14:** add qwen, deepseek, gemma, phi and all possible opensource models based on popularity, evals index, and plan your repo writing stage by stage than full at one go, make sure you better the repo by observing more results from the run n experiments so how do u logs for all those? also need to have code for better visualization and comprehension etc.

**P15:** update context.md and prompts.md n my repo; and identify the gaps n possible suggestions, need full observability and traceability to build evidences for all the claims made from these experiments

**P16:** dgx nvidia dell machine with 256 GB RAM, blackwell arch. fix all the gaps in context.md, and update context.md, also revisit the models list and identify what can be understood or benchmark by running those models? justify with enough facts or details for me to understand it better, im from India so can i be able to access all gpus? whats cost comparison if run it with h100 n b200 in gpu besides local experiments.

**P17:** identify the lesson plan for me to understand all related terminologies and what all im supposed to understand from beginner to expert level?

**P18:** just to make sure to have this single repo to understand all hardware, compiler n model inferences/quantization details in depth from various dimensions, help generate multiple artifacts

**P19:** add to the lesson plan so that i can use my colab pro, and hugging face paid account, dont write any code now, only the lesson plan as md file, add from essentials to experts to understand n do more in the repo context only

**P20:** create sessions and memory md files so that you can start the session later too

**P21:** after your proper logical thinking or reasoning process without having to waste tokens in the next session, update github with sessions and memory md files and stop

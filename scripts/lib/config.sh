# shellcheck shell=bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/lib/config.sh — defaults for the staged TPU benchmark scripts.
#
# Sourced (no shebang, no exec bit). Every value uses `: "${VAR:=default}"` so
# the caller's environment ALWAYS wins — set any of these in your shell to
# override:
#
#     export TPU_NAME=my-bench
#     export TPU_ZONES_PRIMARY="us-east5-a us-east5-b"
#     export GCS_BUCKET=gs://my-other-cache
#     ./scripts/run_all.sh
#
# IMPORTANT REGION NOTE — read before you change ZONES.
# ─────────────────────────────────────────────────────
# ADR-003 and ADR-006 specify `us-central1` as canonical (model cache bucket
# is `gs://rajaghv-tpu-cache` in `us-central1`). However, as of 2026-05, GCP
# no longer offers v5litepod-1 (v5e single-chip) capacity in us-central1.
# Available v5e zones today: us-east5-{a,b,c}, us-west1-c, us-west4-{a,b},
# europe-west4-{a,b}, asia-southeast1-{b,c}.
#
# Trade-off:
#   - Keep bucket in us-central1 (matches ADR-006) → cross-region reads when
#     the TPU lives elsewhere (small egress cost; ~$0.02/GB).
#   - Move bucket to match the TPU region (e.g. us-east5) → diverges from ADR;
#     update DECISIONS.md if you do this.
# This config defaults to the documented ADR (us-central1 bucket) and a
# reasonable v5e zone list (us-east5 first because it has the highest
# preemptible v5e capacity historically).
# ──────────────────────────────────────────────────────────────────────────────

if [[ "${_TPU_BENCH_CONFIG_SOURCED:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi
_TPU_BENCH_CONFIG_SOURCED=1

# ── GCP project ──────────────────────────────────────────────────────────────
# Default: whatever `gcloud config get-value project` returns. Override only if
# you operate multiple projects from one shell.
: "${GCP_PROJECT:=$(gcloud config get-value project 2>/dev/null || echo '')}"

# ── TPU instance ─────────────────────────────────────────────────────────────
: "${TPU_NAME:=tpu-bench-v5e}"               # gcloud resource name
: "${TPU_ACCEL:=v5litepod-1}"                # accelerator type (v5e single chip)
: "${TPU_RUNTIME:=tpu-ubuntu2204-base}"      # VM image family
: "${TPU_PROVISIONING_MODEL:=SPOT}"          # SPOT (preemptible) | STANDARD (on-demand)

# ── Zone lists for multi-zone fallback ────────────────────────────────────────
# Provisioning tries each zone in order until one succeeds. Adding a zone is
# cheap; removing one if you hit policy issues is the right call.
#
# PRIMARY = closest US capacity for v5e single-chip.
# FALLBACK = secondary US zones tried only if primary all return RESOURCE_EXHAUSTED.
# Override either by exporting the var as a space-separated list.
: "${TPU_ZONES_PRIMARY:=us-east5-a us-east5-b us-east5-c}"
: "${TPU_ZONES_FALLBACK:=us-west4-a us-west4-b us-west1-c}"

# ── GCS model + JAX compile cache ────────────────────────────────────────────
# ADR-006 — single-region GCS bucket, gcsfuse mount on TPU VM.
: "${GCS_BUCKET:=gs://rajaghv-tpu-cache}"           # ADR-006 canonical name
: "${GCS_BUCKET_REGION:=us-central1}"               # ADR-006 canonical region
: "${GCS_MOUNT_POINT:=/mnt/gcs-cache}"              # On-VM mount path
: "${HF_HOME_REMOTE:=$GCS_MOUNT_POINT/hf}"          # HF cache root on VM
: "${JAX_COMPILATION_CACHE_DIR_REMOTE:=$GCS_MOUNT_POINT/jax-cache}"  # R4

# ── Tmux session for resilient remote runs (R5) ──────────────────────────────
: "${TMUX_SESSION:=bench}"

# ── Repo path on the TPU VM ──────────────────────────────────────────────────
: "${REMOTE_REPO_DIR:=\$HOME/tpu}"   # NOTE escaped — interpolated on the VM

# ── Local state paths ────────────────────────────────────────────────────────
# Where stage scripts hand off context to each other. Wiped by 70_teardown.sh.
: "${TPU_STATE_DIR:=$REPO_ROOT/.tpu-bench-state}"

# ── HF token handling ────────────────────────────────────────────────────────
# HF_TOKEN is required for gated models (Gemma, PaliGemma, Llama). Stage 1's
# 5-model registry has none gated, so this is OPTIONAL for smoke/quick. The
# 03_validate_hf.sh script handles the conditional.
# Caller exports HF_TOKEN if needed — we never write it to disk.

# ── Cost reference (mirrors README §Cost Reference, RECOMMENDATIONS R8) ──────
# Hourly USD rates — preemptible/spot. Used by 91_predict_cost.sh and
# 90_status.sh to estimate burn.  Verified 2026-05-06.  Update when GCP price
# pages change; mark a `# verified YYYY-MM-DD` next to any line you touch.
declare -gA PRICE_USD_PER_HR=(
    [tpu_v5litepod_1]=0.36          # v5e-1 spot, README §Hardware Landscape
    [tpu_v6e_1]=0.75                # v6e-1 spot
    [gpu_t4]=0.105                  # 1×T4 spot, asia-south1 (n1-standard-4 host extra)
    [gpu_l4]=0.293                  # g2-standard-4 spot all-in (1×L4 + VM)
    [n1_standard_4]=0.0456          # spot, asia-south1
    [pd_balanced_gb_mo]=0.12        # asia-south1
    [external_ip_in_use]=0.005      # uniform globally since 2024-02
)

# Estimated wall-clock minutes per suite on baseline v5e-1 (README §Suites).
declare -gA SUITE_BASELINE_MINUTES=(
    [smoke]=8
    [quick]=50
    [domain]=60
    [arch]=40
    [llm]=120
    [full]=480
)

# Speedup factor relative to v5e-1 baseline. >1 means slower (multiply minutes).
# These are educated guesses; the prediction script flags them as approximate
# and the right thing to do after one real `quick` run is to refit them.
declare -gA DEVICE_SPEEDUP_FACTOR=(
    [tpu_v5litepod_1]=1.0
    [tpu_v6e_1]=0.55
    [gpu_t4]=1.6
    [gpu_l4]=1.0
)

# ── Budget guardrails (R8) ────────────────────────────────────────────────────
# Soft budget for one session. The cost-prediction script (91_) warns if a
# planned suite would exceed this.  GCP-side budget alerts are configured by
# 11_setup_budget.sh.
: "${SESSION_BUDGET_USD:=5.00}"

# ── Echo the resolved config when called as `source config.sh --print` ────────
if [[ "${1:-}" == "--print" ]]; then
    cat <<EOF
GCP_PROJECT          = $GCP_PROJECT
TPU_NAME             = $TPU_NAME
TPU_ACCEL            = $TPU_ACCEL
TPU_RUNTIME          = $TPU_RUNTIME
TPU_PROVISIONING_MODEL = $TPU_PROVISIONING_MODEL
TPU_ZONES_PRIMARY    = $TPU_ZONES_PRIMARY
TPU_ZONES_FALLBACK   = $TPU_ZONES_FALLBACK
GCS_BUCKET           = $GCS_BUCKET
GCS_BUCKET_REGION    = $GCS_BUCKET_REGION
GCS_MOUNT_POINT      = $GCS_MOUNT_POINT
HF_HOME_REMOTE       = $HF_HOME_REMOTE
JAX_COMPILATION_CACHE_DIR_REMOTE = $JAX_COMPILATION_CACHE_DIR_REMOTE
TMUX_SESSION         = $TMUX_SESSION
REMOTE_REPO_DIR      = $REMOTE_REPO_DIR
TPU_STATE_DIR        = $TPU_STATE_DIR
SESSION_BUDGET_USD   = $SESSION_BUDGET_USD
EOF
fi

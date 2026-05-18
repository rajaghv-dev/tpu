#!/usr/bin/env bash
# PAID: This script provisions a Cloud TPU VM (billing starts when state=READY)
#       and runs the matmul experiment end-to-end. It does NOT auto-delete the
#       VM so you can browse Grafana afterwards — run ./delete_tpu_vm.sh when
#       finished, or pass --auto-delete to delete on success.
# ──────────────────────────────────────────────────────────────────────────────
# run_all.sh — full real-TPU run from a clean checkout.
#
# Steps (each one is idempotent / skips if already done):
#   1. Verify gcloud + docker + auth are ready
#   2. Create the Cloud Monitoring service account (if missing)
#   3. Bring up the local Grafana stack (if not running)
#   4. Create the TPU VM (if missing)
#   5. Install JAX on the VM (always — it's a no-op if already installed)
#   6. Run the matmul experiment under XLA dumps + jax.profiler
#   7. Pull artifacts back locally
#   8. Print the Grafana URL + dashboards + cleanup reminder
#
# Usage:
#   ./run_all.sh                  # full e2e, prompts before paid steps
#   ./run_all.sh --yes            # skip all confirmation prompts
#   ./run_all.sh --auto-delete    # delete the TPU VM after a successful run
#   ./run_all.sh --n-steps 50 --hidden-size 1024 --precision bf16
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=_env.sh
source "$SCRIPT_DIR/_env.sh"

log() { printf '[run_all] %s\n' "$*"; }
err() { printf '[run_all] ERROR: %s\n' "$*" >&2; exit 1; }

ASSUME_YES=0
AUTO_DELETE=0
N_STEPS=10
BATCH_SIZE=32
HIDDEN_SIZE=512
PRECISION="bf16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y) ASSUME_YES=1; shift ;;
        --auto-delete) AUTO_DELETE=1; shift ;;
        --n-steps) N_STEPS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | head -25 | sed 's/^# //'
            exit 0
            ;;
        *) err "unknown arg: $1" ;;
    esac
done

YES_FLAG=""
[[ "$ASSUME_YES" -eq 1 ]] && YES_FLAG="--yes"

confirm() {
    [[ "$ASSUME_YES" -eq 1 ]] && return 0
    read -r -p "$1 [y/N] " ans
    case "${ans:-}" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

# ── 1. Pre-flight ─────────────────────────────────────────────────────────────
log "step 1/8 — pre-flight checks"
command -v gcloud >/dev/null || err "gcloud not found. Install: brew install --cask google-cloud-sdk"
command -v docker >/dev/null || err "docker not found. Install Docker Desktop."
docker info >/dev/null 2>&1 || err "docker daemon not running. Start Docker Desktop."
gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q '@' \
    || err "no active gcloud account. Run: gcloud auth login"

# Platform detection — affects stackdriver-exporter (amd64-only image).
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
HOST_ARCH_RAW="$(uname -m)"
case "$HOST_ARCH_RAW" in
    x86_64|amd64)   HOST_ARCH="amd64" ;;
    arm64|aarch64)  HOST_ARCH="arm64" ;;
    *)              HOST_ARCH="$HOST_ARCH_RAW" ;;
esac
log "  host: $HOST_OS/$HOST_ARCH"

if [[ "$HOST_ARCH" == "amd64" ]]; then
    log "  stackdriver-exporter: native amd64 (no emulation)"
elif [[ "$HOST_ARCH" == "arm64" ]]; then
    # On Apple Silicon, Docker Desktop uses Rosetta 2 for amd64 emulation.
    if [[ "$HOST_OS" == "darwin" ]]; then
        if ! /usr/bin/pgrep -q oahd 2>/dev/null && ! arch -x86_64 true 2>/dev/null; then
            log "  WARN: Rosetta 2 may not be installed. If stackdriver-exporter"
            log "        fails to start, run: softwareupdate --install-rosetta --agree-to-license"
        fi
        log "  stackdriver-exporter: linux/amd64 under Rosetta 2 (image is amd64-only)"
    else
        log "  stackdriver-exporter: linux/amd64 under QEMU emulation (image is amd64-only)"
        log "  hint: ensure binfmt is registered — docker run --privileged --rm tonistiigi/binfmt --install all"
    fi
else
    log "  WARN: unknown host arch '$HOST_ARCH_RAW' — stackdriver-exporter may fail to start"
fi
log "  gcloud OK, docker OK, auth OK"

# Enable APIs (idempotent — no-op if already enabled)
log "  ensuring TPU + Monitoring APIs are enabled on $PROJECT_ID ..."
gcloud services enable tpu.googleapis.com monitoring.googleapis.com \
    --project="$PROJECT_ID" >/dev/null

# ── 2. Service account for Cloud Monitoring ──────────────────────────────────
SA_NAME="cloud-tpu-lab-monitoring"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SA_KEY="${HOME}/.config/gcloud/cloud_tpu_lab_sa.json"

log "step 2/8 — Cloud Monitoring service account"
if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" >/dev/null 2>&1; then
    log "  SA $SA_EMAIL already exists — skip create"
else
    log "  creating SA $SA_EMAIL ..."
    gcloud iam service-accounts create "$SA_NAME" --project="$PROJECT_ID" \
        --display-name="cloud_tpu_lab local stackdriver-exporter"
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="roles/monitoring.viewer" >/dev/null
fi

if [[ -f "$SA_KEY" ]]; then
    log "  SA key already at $SA_KEY — skip"
else
    log "  creating SA key at $SA_KEY ..."
    mkdir -p "$(dirname "$SA_KEY")"
    gcloud iam service-accounts keys create "$SA_KEY" \
        --iam-account="$SA_EMAIL"
    chmod 600 "$SA_KEY"
fi

# ── 3. Local Grafana stack ───────────────────────────────────────────────────
log "step 3/8 — local observability stack"
if docker compose -f "$REPO_ROOT/observability/docker-compose.yml" ps --services --filter "status=running" 2>/dev/null | grep -q grafana; then
    log "  stack already running — skip"
else
    log "  starting Prometheus + Grafana + Loki + Tempo + stackdriver-exporter ..."
    (cd "$REPO_ROOT/observability" && docker compose up -d)
    log "  waiting 5s for services to settle ..."
    sleep 5
fi
log "  Grafana → http://localhost:3000  (admin / admin)"

# ── 4. TPU VM ────────────────────────────────────────────────────────────────
log "step 4/8 — TPU VM"
EXISTING=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" \
    --format='value(state)' 2>/dev/null || true)
if [[ -n "$EXISTING" ]]; then
    log "  TPU '$TPU_NAME' exists (state=$EXISTING) — skip create"
else
    log "  TPU '$TPU_NAME' missing — will create ($ACCELERATOR_TYPE in $ZONE)"
    confirm "  This is PAID. Create now?" || err "aborted before VM creation."
    "$SCRIPT_DIR/create_tpu_vm.sh" --yes
fi

# ── 5. Install JAX on the VM ─────────────────────────────────────────────────
log "step 5/8 — install JAX + tensorboard on the VM"
"$SCRIPT_DIR/install_jax_tpu.sh"

# ── 6. Run the experiment ───────────────────────────────────────────────────
log "step 6/8 — run matmul experiment (steps=$N_STEPS, N=$HIDDEN_SIZE, $PRECISION)"
"$SCRIPT_DIR/run_real_demo.sh" $YES_FLAG \
    --n-steps "$N_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-size "$HIDDEN_SIZE" \
    --precision "$PRECISION"

# ── 7. Collect artifacts ────────────────────────────────────────────────────
log "step 7/8 — pull artifacts back to artifacts/from_vm/"
"$SCRIPT_DIR/collect_artifacts.sh"

# ── 8. Done ──────────────────────────────────────────────────────────────────
log "step 8/8 — done"
cat <<DONE

============================================================
  RUN COMPLETE
------------------------------------------------------------
  Local artifacts : $REPO_ROOT/artifacts/from_vm/
  Grafana         : http://localhost:3000  (admin / admin)
    workload-level: cloud_tpu_overview, compile_and_runtime,
                    hbm_memory, cost_performance, debugging
    GCP infra     : cloud_tpu_gcp_metrics
  Perfetto trace  : drag any artifacts/from_vm/*/run_*.json
                    into https://ui.perfetto.dev

  TPU VM '$TPU_NAME' is STILL RUNNING and still billing.
DONE

if [[ "$AUTO_DELETE" -eq 1 ]]; then
    log "auto-delete enabled — tearing down TPU VM ..."
    "$SCRIPT_DIR/delete_tpu_vm.sh"
else
    cat <<REMINDER
  Cleanup when finished:
      $SCRIPT_DIR/delete_tpu_vm.sh
============================================================
REMINDER
fi

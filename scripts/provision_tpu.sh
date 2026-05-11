#!/usr/bin/env bash
# Provision a (preemptible|spot) Cloud TPU VM, copy the repo, install deps,
# arm a self-teardown safety net, and run hello-TPU.
#
# Defaults match DECISIONS.md / ADR-003: v5e-1 preemptible @ ~$0.36/hr in us-west4-a.
# Defaults must stay in sync with teardown_tpu.sh.
#
# Usage:
#   ./scripts/provision_tpu.sh [TPU_NAME] [ZONE] [ACCELERATOR_TYPE] [flags...]
#
# Flags:
#   --spot                Use --spot instead of --preemptible (may be 10–20% cheaper,
#                         unconfirmed for v5e at this time).
#   --preemptible         Force preemptible (default; matches ADR-003).
#   --max-runtime=DURATION  Schedule self-shutdown on the VM after DURATION.
#                         Accepts e.g. "2h", "90m", "30m", or "none" to disable.
#                         Default: 2h. Implemented via `sudo shutdown -h +MINUTES`
#                         scheduled on the VM at the end of provisioning.
#   --dry-run             Print every command that would run (create, copy,
#                         install, at-job) without executing. Mirrors
#                         gcp_bootstrap.sh --check style. Exits 0.
#   --help / -h           Show this help and exit.
#
# Env overrides (optional GCS cache hooks — see Agent B's cache_wheels.sh):
#   WHEEL_CACHE_URL     e.g. gs://<bucket>/wheels/    Pre-built wheel set.
#   HF_MODEL_CACHE_URL  e.g. gs://<bucket>/hf-cache/  HF Hub cache mirror.
# If unset OR the GCS path is not reachable, falls back to fresh install +
# fresh HF downloads (legacy behavior).

set -euo pipefail

# ── Defaults (positional args 1..3) ─────────────────────────────────────
TPU_NAME=""
ZONE=""
ACCEL=""
RUNTIME="tpu-ubuntu2204-base"

# Flag-controlled state.
PROVISION_MODE="preemptible"   # or "spot"
MAX_RUNTIME="2h"
DRY_RUN=false

# GCS cache hooks (env-var protocol with Agent B).
WHEEL_CACHE_URL="${WHEEL_CACHE_URL:-}"
HF_MODEL_CACHE_URL="${HF_MODEL_CACHE_URL:-}"

# ── Log helpers (match gcp_bootstrap.sh) ────────────────────────────────
log()   { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
ok()    { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn()  { printf "  \033[33m!\033[0m %s\n" "$*"; }
err()   { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; }

usage() {
  # Print the header comment block (everything up to the first blank-after-comments).
  awk '
    NR==1 { next }                       # skip the shebang
    /^[^#]/ { exit }                     # stop at first non-comment line
    { sub(/^# ?/, ""); print }
  ' "$0"
  exit "${1:-0}"
}

# ── Arg parsing ─────────────────────────────────────────────────────────
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spot)         PROVISION_MODE="spot"; shift ;;
    --preemptible)  PROVISION_MODE="preemptible"; shift ;;
    --max-runtime=*) MAX_RUNTIME="${1#*=}"; shift ;;
    --max-runtime)  MAX_RUNTIME="${2:?--max-runtime needs a value}"; shift 2 ;;
    --dry-run)      DRY_RUN=true; shift ;;
    -h|--help)      usage 0 ;;
    --*)            err "Unknown flag: $1"; usage 2 ;;
    *)              POSITIONAL+=("$1"); shift ;;
  esac
done

TPU_NAME="${POSITIONAL[0]:-tpu-demo}"
ZONE="${POSITIONAL[1]:-us-west4-a}"
ACCEL="${POSITIONAL[2]:-v5litepod-1}"

# ── Wall-time + cost telemetry ──────────────────────────────────────────
START_TS=$(date +%s)
HOURLY_RATE="0.36"   # v5e-1 preemptible nominal. Spot may differ.

elapsed() {
  local now=$(( $(date +%s) - START_TS ))
  printf "%02d:%02d" "$((now/60))" "$((now%60))"
}

cost_so_far() {
  local secs=$(( $(date +%s) - START_TS ))
  awk -v s="$secs" -v r="$HOURLY_RATE" 'BEGIN{printf "%.3f", (s/3600.0)*r}'
}

step() {
  printf "\n\033[1;34m▶ [%s] %s  (cost so far: \$%s)\033[0m\n" \
    "$(elapsed)" "$*" "$(cost_so_far)"
}

# ── DRY-RUN-aware command runner ────────────────────────────────────────
# Prints the command always; only executes when DRY_RUN=false.
# NOTE: pass args directly ("$@") — do NOT eval. eval re-parses, which
# breaks the inner quotes in --command="..." values and causes remote-side
# vars (e.g. $JAX_COMPILATION_CACHE_DIR) to be expanded locally where they
# are unset (set -u then trips). All callers use straight argv, no pipes.
run() {
  printf "  \033[2m\$ %s\033[0m\n" "$*"
  if ! $DRY_RUN; then
    "$@"
  fi
}

# ── Repo-state capture (used for git-clone-first copy strategy) ─────────
log "Capturing local repo state"
REMOTE_URL=""
BRANCH=""
SHA=""
HAVE_GIT_REMOTE=false
DETACHED=false

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if REMOTE_URL=$(git remote get-url origin 2>/dev/null); then
    HAVE_GIT_REMOTE=true
    ok "origin: $REMOTE_URL"
  else
    warn "no 'origin' remote — will fall back to tar-with-excludes"
  fi
  BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
  if [[ "$BRANCH" == "HEAD" || -z "$BRANCH" ]]; then
    DETACHED=true
    warn "detached HEAD — will fall back to tar-with-excludes"
  else
    ok "branch: $BRANCH"
  fi
  SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  ok "sha: $SHA"
else
  warn "not inside a git work tree — will fall back to tar-with-excludes"
fi

# Local pre-flight: count uncommitted files so we can report N at the end.
UNCOMMITTED_LIST=""
UNCOMMITTED_COUNT=0
if $HAVE_GIT_REMOTE && ! $DETACHED; then
  if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    UNCOMMITTED_LIST=$(
      { git ls-files --others --exclude-standard; git diff --name-only; } \
        | sort -u
    )
  else
    # No staged/working changes vs HEAD, but there can still be untracked files.
    UNCOMMITTED_LIST=$(git ls-files --others --exclude-standard | sort -u)
  fi
  UNCOMMITTED_COUNT=$(printf "%s" "$UNCOMMITTED_LIST" | sed '/^$/d' | wc -l | tr -d ' ')
  ok "uncommitted/untracked files: $UNCOMMITTED_COUNT"
fi

# ── Plan summary ────────────────────────────────────────────────────────
log "Provisioning plan"
cat <<EOF
  TPU_NAME       = $TPU_NAME
  ZONE           = $ZONE
  ACCEL          = $ACCEL
  RUNTIME        = $RUNTIME
  MODE           = --$PROVISION_MODE
  MAX_RUNTIME    = $MAX_RUNTIME  (self-shutdown via 'sudo shutdown -h')
  WHEEL_CACHE    = ${WHEEL_CACHE_URL:-<unset, will pip-install fresh>}
  HF_CACHE       = ${HF_MODEL_CACHE_URL:-<unset, will download fresh>}
  COPY_STRATEGY  = $( $HAVE_GIT_REMOTE && ! $DETACHED \
                       && echo "git clone --depth=1 + tar overlay ($UNCOMMITTED_COUNT files)" \
                       || echo "full tar-with-excludes (no usable git remote)" )
  DRY_RUN        = $DRY_RUN
EOF

if $DRY_RUN; then
  warn "DRY-RUN mode — no gcloud commands will execute."
fi

# Common shell prelude shared by every remote `--command=` block. It exports
# the XLA compile cache dir so subsequent runs on this VM hit the on-disk cache
# (per-VM-lifetime; cross-provision persistence requires GCS sync — out of scope).
REMOTE_PRELUDE='export JAX_COMPILATION_CACHE_DIR="$HOME/xla-cache" && mkdir -p "$JAX_COMPILATION_CACHE_DIR"'

# ── Create or attach to the VM (idempotent: skip create if it already exists) ──
# Rationale: if a previous run failed mid-script, the VM is up but unconfigured.
# Re-running provision_tpu.sh should resume from where it failed, not abort with
# ALREADY_EXISTS. All downstream steps are idempotent (HF token overwrites,
# git clone has `rm -rf` first, pip install skips installed, otelcol install
# has an existence guard, shutdown -h cancels+reschedules prior).
if gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" >/dev/null 2>&1; then
  step "VM $TPU_NAME already exists in $ZONE — skipping create (resuming setup)"
else
  step "Creating TPU VM ($ACCEL, --$PROVISION_MODE) — \$${HOURLY_RATE}/hr starts now"
  CREATE_CMD=(
    gcloud compute tpus tpu-vm create "$TPU_NAME"
    --zone="$ZONE"
    --accelerator-type="$ACCEL"
    --version="$RUNTIME"
    "--${PROVISION_MODE}"
  )
  run "${CREATE_CMD[@]}"
fi

step "VM ready"

# ── Persist JAX_COMPILATION_CACHE_DIR in the VM's ~/.profile ────────────
# Per Fix 4 (Tier 2 #6): make the env var sticky for any future SSH session
# on this VM. Idempotent — only appended once per VM lifetime.
step "Persisting XLA compile-cache env in ~/.profile"
PROFILE_SNIPPET='# tpu-bench: XLA compile cache (Tier 2 #6)\nexport JAX_COMPILATION_CACHE_DIR="$HOME/xla-cache"\nmkdir -p "$JAX_COMPILATION_CACHE_DIR"'
run gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="'grep -q JAX_COMPILATION_CACHE_DIR \$HOME/.profile 2>/dev/null || printf \"%b\\n\" \"$PROFILE_SNIPPET\" >> \$HOME/.profile'"

# ── Propagate HuggingFace token (gated models: Gemma/LLaMA/PaliGemma) ───
# Priority: GCP Secret Manager (preferred — no laptop→VM transit of the
# secret) > ~/.cache/huggingface/token > ~/.huggingface/token > HF_TOKEN env.
# Token is NEVER printed or logged. See scripts/setup_hf.sh.
HF_TOKEN_VALUE=""
HF_TOKEN_SOURCE=""
if gcloud secrets describe hf-token >/dev/null 2>&1; then
  HF_TOKEN_VALUE=$(gcloud secrets versions access latest --secret=hf-token 2>/dev/null \
                   | tr -d '[:space:]' || true)
  [[ -n "$HF_TOKEN_VALUE" ]] && HF_TOKEN_SOURCE="GCP Secret Manager (hf-token)"
fi
if [[ -z "$HF_TOKEN_VALUE" && -f "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN_VALUE=$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")
  HF_TOKEN_SOURCE="~/.cache/huggingface/token"
fi
if [[ -z "$HF_TOKEN_VALUE" && -f "$HOME/.huggingface/token" ]]; then
  HF_TOKEN_VALUE=$(tr -d '[:space:]' < "$HOME/.huggingface/token")
  HF_TOKEN_SOURCE="~/.huggingface/token"
fi
if [[ -z "$HF_TOKEN_VALUE" && -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_VALUE="$HF_TOKEN"
  HF_TOKEN_SOURCE="HF_TOKEN env var"
fi

if [[ -n "$HF_TOKEN_VALUE" ]]; then
  step "Propagating HF token (source: $HF_TOKEN_SOURCE) to VM"
  # Write to a temp file with mode 0600, scp it to the VM, then shred it.
  # The destination on the VM is the HF library's canonical token path —
  # transformers / huggingface_hub auto-read it. No env-var leak in shell history.
  TMP_TOKEN=$(mktemp)
  chmod 600 "$TMP_TOKEN"
  trap 'shred -u "$TMP_TOKEN" 2>/dev/null || rm -f "$TMP_TOKEN"' EXIT
  printf "%s" "$HF_TOKEN_VALUE" > "$TMP_TOKEN"
  # Prepare the VM dir first (scp can't mkdir).
  run gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
    --command="'mkdir -p \$HOME/.cache/huggingface \$HOME/.huggingface && chmod 700 \$HOME/.cache/huggingface \$HOME/.huggingface'"
  run gcloud compute tpus tpu-vm scp "$TMP_TOKEN" \
    "$TPU_NAME":~/.cache/huggingface/token --zone="$ZONE"
  run gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
    --command="'chmod 600 \$HOME/.cache/huggingface/token && cp \$HOME/.cache/huggingface/token \$HOME/.huggingface/token && chmod 600 \$HOME/.huggingface/token'"
  ok "HF token installed at \$HOME/.cache/huggingface/token (mode 0600)"
else
  warn "No HF token found locally or in Secret Manager — gated models (Gemma/LLaMA/PaliGemma) will fail."
  warn "  Set up once via: ./scripts/setup_hf.sh"
fi

# ── Copy repo: hybrid Option C (git clone + tar overlay) ────────────────
step "Copying repo to ~/tpu-examples"

if $HAVE_GIT_REMOTE && ! $DETACHED; then
  ok "Strategy: git clone --depth=1 --branch=$BRANCH (only tracked files, no .tpu/.git history/.claude)"
  CLONE_CMD="$REMOTE_PRELUDE && rm -rf \$HOME/tpu-examples && git clone --depth=1 --branch=$BRANCH '$REMOTE_URL' \$HOME/tpu-examples"
  run gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
    --command="\"$CLONE_CMD\""

  if [[ "$UNCOMMITTED_COUNT" -gt 0 ]]; then
    ok "Overlaying $UNCOMMITTED_COUNT uncommitted/untracked file(s) via tar (with excludes)"
    # Build a NUL-delimited list piped to `tar -T -` so paths with spaces are safe,
    # and so we never pass an empty arg list to tar.
    if $DRY_RUN; then
      printf "  \033[2m\$ tar --exclude=.tpu --exclude=.git --exclude=.claude \\\\\n"
      printf "       --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.pyc' \\\\\n"
      printf "       --exclude=results/otel --exclude=results/run_logs \\\\\n"
      printf "       --exclude='results/*.log' --exclude='*.tar.gz' --exclude=otelcol-contrib \\\\\n"
      printf "       -czf - -T <(printf '%%s\\\\0' <files>) --null \\\\\n"
      printf "    | gcloud compute tpus tpu-vm ssh %s --zone=%s \\\\\n" "$TPU_NAME" "$ZONE"
      printf "        --command='cd \$HOME/tpu-examples && tar -xzf -'\033[0m\n"
    else
      # Filter the file list to existing paths (deleted files would fail tar).
      EXISTING=$(printf "%s\n" "$UNCOMMITTED_LIST" | while IFS= read -r f; do
        [[ -n "$f" && -e "$f" ]] && printf "%s\0" "$f"
      done)
      if [[ -z "$EXISTING" ]]; then
        warn "All listed uncommitted files were deleted locally — nothing to overlay."
      else
        printf "%s" "$EXISTING" \
          | tar --null --files-from=- \
                --exclude='.tpu' --exclude='.git' --exclude='.claude' \
                --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc' \
                --exclude='results/otel' --exclude='results/run_logs' \
                --exclude='results/*.log' --exclude='*.tar.gz' --exclude='otelcol-contrib' \
                -czf - \
          | gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
              --command='cd $HOME/tpu-examples && tar -xzf -'
      fi
    fi
  else
    ok "No uncommitted files — clone alone is the complete copy."
  fi

  step "Repo synced — cloned ${REMOTE_URL}@${BRANCH} (${SHA}); overlaid ${UNCOMMITTED_COUNT} uncommitted file(s)"
else
  # Fallback: full tar with the same excludes. Never the recursive-scp-of-cwd
  # antipattern that copied .tpu/.git/.claude in the original script.
  warn "Falling back to full tar-with-excludes (no git remote or detached HEAD)."
  if $DRY_RUN; then
    printf "  \033[2m\$ tar -C %s -czf - --exclude=.tpu --exclude=.git --exclude=.claude \\\\\n" "$PWD"
    printf "       --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.pyc' \\\\\n"
    printf "       --exclude=results/otel --exclude=results/run_logs \\\\\n"
    printf "       --exclude='results/*.log' --exclude='*.tar.gz' --exclude=otelcol-contrib . \\\\\n"
    printf "    | gcloud compute tpus tpu-vm ssh %s --zone=%s \\\\\n" "$TPU_NAME" "$ZONE"
    printf "        --command='mkdir -p \$HOME/tpu-examples && cd \$HOME/tpu-examples && tar -xzf -'\033[0m\n"
  else
    tar -C "$PWD" \
        --exclude='.tpu' --exclude='.git' --exclude='.claude' \
        --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc' \
        --exclude='results/otel' --exclude='results/run_logs' \
        --exclude='results/*.log' --exclude='*.tar.gz' --exclude='otelcol-contrib' \
        -czf - . \
      | gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
          --command='mkdir -p $HOME/tpu-examples && cd $HOME/tpu-examples && tar -xzf -'
  fi
  step "Repo synced (fallback tar mode)"
fi

# ── Optional GCS caches (wheels + HF) ───────────────────────────────────
WHEEL_CACHE_ACTIVE=false
HF_CACHE_ACTIVE=false

if [[ -n "$WHEEL_CACHE_URL" ]]; then
  step "Probing wheel cache: $WHEEL_CACHE_URL"
  if $DRY_RUN; then
    printf "  \033[2m\$ gsutil -q stat '%s**' && pull wheels into /tmp/wheels on VM\033[0m\n" "$WHEEL_CACHE_URL"
    WHEEL_CACHE_ACTIVE=true
  else
    if gsutil -q stat "${WHEEL_CACHE_URL%/}/**" 2>/dev/null \
       || gsutil ls "$WHEEL_CACHE_URL" >/dev/null 2>&1; then
      ok "Wheel cache reachable — will use --find-links"
      WHEEL_CACHE_ACTIVE=true
    else
      warn "Wheel cache URL set but not reachable — falling back to fresh pip install"
    fi
  fi
else
  ok "WHEEL_CACHE_URL unset — fresh pip install (legacy behavior)"
fi

if [[ -n "$HF_MODEL_CACHE_URL" ]]; then
  step "Probing HF model cache: $HF_MODEL_CACHE_URL"
  if $DRY_RUN; then
    printf "  \033[2m\$ gsutil -q stat '%s**' && rsync into ~/.cache/huggingface/hub on VM\033[0m\n" "$HF_MODEL_CACHE_URL"
    HF_CACHE_ACTIVE=true
  else
    if gsutil ls "$HF_MODEL_CACHE_URL" >/dev/null 2>&1; then
      ok "HF cache reachable — will rsync into ~/.cache/huggingface/hub"
      HF_CACHE_ACTIVE=true
    else
      warn "HF_MODEL_CACHE_URL set but not reachable — falling back to fresh downloads"
    fi
  fi
else
  ok "HF_MODEL_CACHE_URL unset — fresh HF downloads (legacy behavior)"
fi

# ── Install Python deps (with optional wheel cache) ─────────────────────
step "Installing Python deps"
if $WHEEL_CACHE_ACTIVE; then
  INSTALL_CMD="$REMOTE_PRELUDE && mkdir -p /tmp/wheels && gsutil -m cp -r '${WHEEL_CACHE_URL%/}/*' /tmp/wheels/ && pip install --quiet --find-links=/tmp/wheels -r \$HOME/tpu-examples/requirements.txt"
else
  INSTALL_CMD="$REMOTE_PRELUDE && pip install --quiet -r \$HOME/tpu-examples/requirements.txt"
fi
run gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="\"$INSTALL_CMD\""

# ── Optional: rsync HF model cache from GCS ─────────────────────────────
if $HF_CACHE_ACTIVE; then
  step "Syncing HF model cache from GCS"
  HF_CMD="mkdir -p \$HOME/.cache/huggingface/hub && gsutil -m rsync -r '$HF_MODEL_CACHE_URL' \$HOME/.cache/huggingface/hub/"
  run gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
    --command="\"$HF_CMD\""
fi

# ── Install OTel collector (unchanged behavior) ─────────────────────────
step "Installing OpenTelemetry Collector (otelcol-contrib v0.105.0)"
OTEL_CMD='
    set -e
    cd $HOME/tpu-examples
    OTELCOL_VERSION=0.105.0
    if [ ! -x ./otelcol-contrib ]; then
      curl -sLO "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTELCOL_VERSION}/otelcol-contrib_${OTELCOL_VERSION}_linux_amd64.tar.gz"
      tar -xzf otelcol-contrib_${OTELCOL_VERSION}_linux_amd64.tar.gz otelcol-contrib
      chmod +x otelcol-contrib
      rm -f otelcol-contrib_${OTELCOL_VERSION}_linux_amd64.tar.gz
    fi
    mkdir -p results/otel
  '
run gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="\"$OTEL_CMD\""

# ── Auto-teardown safety net (Tier 3 #11) ───────────────────────────────
# Convert MAX_RUNTIME to minutes for `shutdown -h +N`. Accepts Nh, Nm, "none".
shutdown_minutes_from() {
  local v="$1"
  case "$v" in
    none|disabled|"") echo ""; return ;;
    *h)               echo $(( ${v%h} * 60 )) ;;
    *m)               echo "${v%m}" ;;
    *[0-9])           echo "$v" ;;   # bare number → minutes
    *) err "Unrecognized --max-runtime value: $v (use e.g. 2h, 90m, or none)"; exit 2 ;;
  esac
}

SHUTDOWN_MIN=$(shutdown_minutes_from "$MAX_RUNTIME")
if [[ -n "$SHUTDOWN_MIN" ]]; then
  step "Arming self-teardown: sudo shutdown -h +${SHUTDOWN_MIN} (max runtime ${MAX_RUNTIME})"
  # We schedule the shutdown directly via `shutdown -h +MIN`. This sits in the
  # init system's timer and cleanly halts the VM, which stops billing. The
  # operator can cancel any time with `sudo shutdown -c` (or by running
  # teardown_tpu.sh, which deletes the VM outright).
  SHUTDOWN_CMD="sudo shutdown -h +${SHUTDOWN_MIN} 'tpu-bench self-teardown after ${MAX_RUNTIME}'"
  run gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
    --command="\"$SHUTDOWN_CMD\""
else
  warn "Self-teardown DISABLED (--max-runtime=none). Don't forget teardown_tpu.sh!"
fi

# ── hello-TPU smoke check ───────────────────────────────────────────────
step "Running hello-TPU check"
HELLO_CMD="$REMOTE_PRELUDE && python \$HOME/tpu-examples/01_hello_tpu/hello_tpu.py"
run gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="\"$HELLO_CMD\""

# ── Done ────────────────────────────────────────────────────────────────
step "Done"

# Format SHUTDOWN_MIN as HHhMMm for the closing banner.
fmt_runtime() {
  local m="$1"
  [[ -z "$m" ]] && { echo "disabled"; return; }
  printf "%dh%02dm" "$((m/60))" "$((m%60))"
}
RUNTIME_PRETTY=$(fmt_runtime "$SHUTDOWN_MIN")

echo
if [[ -n "$SHUTDOWN_MIN" ]]; then
  printf "\033[1;33m⚠ VM will self-terminate in %s (override: --max-runtime).\033[0m\n" "$RUNTIME_PRETTY"
  printf "  Run \033[1mteardown_tpu.sh\033[0m sooner to stop billing now:\n"
  printf "    ./scripts/teardown_tpu.sh %s %s\n" "$TPU_NAME" "$ZONE"
  printf "  Or cancel just the self-shutdown (keep VM alive):\n"
  printf "    gcloud compute tpus tpu-vm ssh %s --zone=%s --command='sudo shutdown -c'\n" "$TPU_NAME" "$ZONE"
else
  printf "\033[1;31m⚠ Self-teardown disabled — VM will run until you delete it.\033[0m\n"
  printf "  Teardown:  ./scripts/teardown_tpu.sh %s %s\n" "$TPU_NAME" "$ZONE"
fi

cat <<EOF

Wheel cache:     $( $WHEEL_CACHE_ACTIVE && echo "USED ($WHEEL_CACHE_URL)" || echo "not used" )
HF model cache:  $( $HF_CACHE_ACTIVE   && echo "USED ($HF_MODEL_CACHE_URL)" || echo "not used" )
Total elapsed:   $(elapsed)   Estimated cost: \$$(cost_so_far)

Next steps:
  Smoke benchmark:  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \\
                      --command='cd tpu-examples && PYTHONPATH=. python benchmarks/harness.py --suite smoke --device tpu_v5e1'

  OTel workflow (see DECISIONS.md ADR-016):
  1. Start OTel collector on TPU (separate SSH session):
       gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \\
         --command='cd tpu-examples && OUT_FILE=results/otel/\$(date +%s).jsonl ./otelcol-contrib --config infra/otelcol-tpu-config.yaml'
  2. Run benchmark with OTel enabled (in main SSH session):
       TPU_BENCH_OTEL=otlp PYTHONPATH=. python benchmarks/harness.py --suite smoke --device tpu_v5e1
  3. Stop the OTel collector (Ctrl+C in its SSH session).
  4. From your laptop:  ./scripts/otel_collect.sh && ./scripts/otel_view.sh

TIP: To avoid passphrase prompts on every gcloud ssh, run once per shell:
       eval \$(ssh-agent) && ssh-add ~/.ssh/google_compute_engine

Teardown (stop billing now):  ./scripts/teardown_tpu.sh $TPU_NAME $ZONE
EOF

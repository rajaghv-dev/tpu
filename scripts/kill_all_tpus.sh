#!/usr/bin/env bash
# Emergency: list and delete ALL TPU VMs in the current project across all zones.
# Use when you've forgotten which zone a VM is in, or want a guaranteed-zero state.
#
# Usage:
#   ./scripts/kill_all_tpus.sh              # list + confirm before delete
#   ./scripts/kill_all_tpus.sh --dry-run    # list only, no delete
#   ./scripts/kill_all_tpus.sh --force      # delete without confirmation
#
# Cost rationale: a v5e-1 preemptible left running = $0.36/hr × 24 × 30 ≈ $259/mo.
# This script is your panic button.

set -euo pipefail

MODE="confirm"
case "${1:-}" in
  --dry-run) MODE="dry-run" ;;
  --force)   MODE="force"   ;;
  "")        MODE="confirm" ;;
  *)
    echo "Usage: $0 [--dry-run|--force]" >&2
    exit 2
    ;;
esac

PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "✗ No project configured. Run: gcloud config set project <id>" >&2
  exit 1
fi

echo "▶ Project: $PROJECT_ID"
echo "▶ Scanning all TPU-capable zones for running VMs ..."

ZONES=$(gcloud compute tpus locations list --format="value(name)" 2>/dev/null || true)
if [[ -z "$ZONES" ]]; then
  echo "  (could not list TPU zones — Cloud TPU API may not be enabled)"
  exit 1
fi

# Collect everything first so we can show a single summary table.
FOUND=()
while IFS= read -r zone; do
  [[ -z "$zone" ]] && continue
  while IFS=$'\t' read -r name state accel; do
    [[ -z "$name" ]] && continue
    FOUND+=("$zone	$name	$state	$accel")
  done < <(gcloud compute tpus tpu-vm list \
              --zone="$zone" \
              --format="value(name,state,acceleratorType)" 2>/dev/null || true)
done <<<"$ZONES"

if [[ ${#FOUND[@]} -eq 0 ]]; then
  echo "✓ No TPU VMs found in $PROJECT_ID. You're at \$0/hr."
  exit 0
fi

printf "\n  %-20s %-25s %-15s %s\n" "ZONE" "NAME" "STATE" "ACCEL"
printf "  %-20s %-25s %-15s %s\n" "----" "----" "-----" "-----"
for line in "${FOUND[@]}"; do
  IFS=$'\t' read -r zone name state accel <<<"$line"
  printf "  %-20s %-25s %-15s %s\n" "$zone" "$name" "$state" "$accel"
done

# Rough burn rate (preemptible v5e-1 only — others vary).
COUNT=${#FOUND[@]}
EST_PER_HR=$(awk -v n="$COUNT" 'BEGIN{printf "%.2f", n*0.36}')
echo
echo "  ${COUNT} VM(s) — burning ≈ \$${EST_PER_HR}/hr if all are preemptible v5e-1"

if [[ "$MODE" == "dry-run" ]]; then
  echo
  echo "  Dry-run only — nothing deleted."
  exit 0
fi

if [[ "$MODE" == "confirm" ]]; then
  echo
  read -r -p "Delete all ${COUNT} VM(s)? [type 'yes' to confirm]: " ANSWER
  [[ "$ANSWER" == "yes" ]] || { echo "Aborted."; exit 1; }
fi

echo
echo "▶ Deleting ..."
FAILED=0
for line in "${FOUND[@]}"; do
  IFS=$'\t' read -r zone name _state _accel <<<"$line"
  if gcloud compute tpus tpu-vm delete "$name" --zone="$zone" --quiet 2>&1; then
    echo "  ✓ deleted $name (zone=$zone)"
  else
    echo "  ✗ FAILED to delete $name (zone=$zone)" >&2
    FAILED=$((FAILED+1))
  fi
done

echo
if [[ "$FAILED" -eq 0 ]]; then
  echo "✓ All TPU VMs deleted. Billing has stopped."
else
  echo "! $FAILED deletion(s) failed — re-run to retry."
  exit 1
fi

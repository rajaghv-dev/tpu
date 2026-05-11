#!/usr/bin/env bash
# Pull OTLP-JSON traces/metrics and runs.jsonl back from a TPU VM.
# See DECISIONS.md ADR-016 (local-only OTel + Grafana, supersedes ADR-015).
# Usage: ./scripts/otel_collect.sh [TPU_NAME] [ZONE]

set -euo pipefail

TPU_NAME="${1:-tpu-demo}"
ZONE="${2:-us-west4-a}"

mkdir -p ./results/otel

echo "Pulling results/otel/ from ${TPU_NAME} (${ZONE}) ..."
gcloud compute tpus tpu-vm scp --recurse \
  "$TPU_NAME":~/tpu-examples/results/otel/* \
  ./results/otel/ \
  --zone="$ZONE"

echo "Pulling results/runs.jsonl (overwrites local copy; use 'git diff' to inspect) ..."
gcloud compute tpus tpu-vm scp \
  "$TPU_NAME":~/tpu-examples/results/runs.jsonl \
  ./results/runs.jsonl \
  --zone="$ZONE" || echo "  (runs.jsonl not present on TPU yet — skipping)"

FILE_COUNT=$(find ./results/otel -type f -name '*.jsonl' 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh ./results/otel 2>/dev/null | awk '{print $1}')

echo
echo "Pulled ${FILE_COUNT} OTLP-JSON file(s), total ${TOTAL_SIZE} in ./results/otel/"
echo
echo "Next: ./scripts/otel_view.sh   (starts local Grafana stack)"

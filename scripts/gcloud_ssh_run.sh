#!/usr/bin/env bash
# SSH into a TPU VM and run a specific example.
# Usage: ./scripts/gcloud_ssh_run.sh <TPU_NAME> <ZONE> <EXAMPLE_DIR> [SCRIPT_ARGS...]
# Example: ./scripts/gcloud_ssh_run.sh tpu-demo us-central1-a 02_mnist_classification

set -euo pipefail

TPU_NAME="${1:?Usage: $0 <TPU_NAME> <ZONE> <EXAMPLE_DIR> [args]}"
ZONE="${2:?}"
EXAMPLE="${3:?}"
shift 3
ARGS="${*:-}"

CMD="cd ~/tpu-examples && python ${EXAMPLE}/train.py ${ARGS}"

echo "Running '${EXAMPLE}' on ${TPU_NAME} (${ZONE}) ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --command="$CMD"

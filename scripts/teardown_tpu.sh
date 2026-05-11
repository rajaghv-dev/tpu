#!/usr/bin/env bash
# Delete a Cloud TPU VM to stop billing.
# Usage: ./scripts/teardown_tpu.sh [TPU_NAME] [ZONE]

set -euo pipefail

TPU_NAME="${1:-tpu-demo}"
ZONE="${2:-us-west4-a}"

echo "Deleting TPU VM: $TPU_NAME in $ZONE ..."
gcloud compute tpus tpu-vm delete "$TPU_NAME" \
  --zone="$ZONE" \
  --quiet

echo "TPU deleted. Billing has stopped."

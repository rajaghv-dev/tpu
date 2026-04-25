#!/usr/bin/env bash
# Provision a Cloud TPU VM, install deps, and run the hello-TPU check.
# Usage: ./scripts/provision_tpu.sh [TPU_NAME] [ZONE] [ACCELERATOR_TYPE]

set -euo pipefail

TPU_NAME="${1:-tpu-demo}"
ZONE="${2:-us-central1-a}"
ACCEL="${3:-v3-8}"
RUNTIME="tpu-vm-base"

echo "Creating TPU VM: $TPU_NAME ($ACCEL) in $ZONE ..."
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --zone="$ZONE" \
  --accelerator-type="$ACCEL" \
  --version="$RUNTIME"

echo "Copying repo ..."
gcloud compute tpus tpu-vm scp --recurse "$(pwd)" \
  "$TPU_NAME":~/tpu-examples --zone="$ZONE"

echo "Installing dependencies ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="pip install --quiet -r ~/tpu-examples/requirements.txt"

echo "Running hello-TPU check ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="python ~/tpu-examples/01_hello_tpu/hello_tpu.py"

echo "Done. SSH in with:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"

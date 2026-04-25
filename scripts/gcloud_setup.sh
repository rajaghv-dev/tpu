#!/usr/bin/env bash
# One-time gcloud project and API setup for Cloud TPU usage.
# Usage: ./scripts/gcloud_setup.sh <PROJECT_ID>

set -euo pipefail

PROJECT_ID="${1:?Usage: $0 <PROJECT_ID>}"

echo "Setting project: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

echo "Enabling required APIs ..."
gcloud services enable \
  compute.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com

echo "Verifying TPU quota (us-central1-a, v3-8) ..."
gcloud compute tpus list --zone=us-central1-a 2>/dev/null || true

echo "Done. You can now run ./scripts/provision_tpu.sh"

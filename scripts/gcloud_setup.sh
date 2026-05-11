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

echo "Listing existing TPUs in us-west4-a (target zone for v5e-1) ..."
gcloud compute tpus list --zone=us-west4-a 2>/dev/null || true

cat <<'EOF'

Done. Before running ./scripts/provision_tpu.sh:
  1. Request quota: IAM & Admin → Quotas → filter "Preemptible TPU v5 Lite"
     → request ≥1 in us-west4 (other v5e zones: us-central1-a, europe-west4-a,
       asia-southeast1-a).
  2. Verify quota approved (can take hours-days).
  3. Then: ./scripts/provision_tpu.sh
EOF

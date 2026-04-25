#!/usr/bin/env bash
# Upload a local dataset directory to a GCS bucket.
# Usage: ./scripts/gcloud_upload_data.sh <LOCAL_DIR> <GCS_BUCKET> [PREFIX]
# Example: ./scripts/gcloud_upload_data.sh ./data/imagenet gs://my-bucket imagenet

set -euo pipefail

LOCAL_DIR="${1:?Usage: $0 <LOCAL_DIR> <GCS_BUCKET> [PREFIX]}"
BUCKET="${2:?}"
PREFIX="${3:-data}"

echo "Uploading ${LOCAL_DIR} → ${BUCKET}/${PREFIX}/ ..."
gsutil -m cp -r "$LOCAL_DIR" "${BUCKET}/${PREFIX}/"
echo "Done. Access path: ${BUCKET}/${PREFIX}/$(basename "$LOCAL_DIR")"

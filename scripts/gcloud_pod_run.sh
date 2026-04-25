#!/usr/bin/env bash
# Launch the multi-host training example across all workers of a TPU pod.
# Usage: ./scripts/gcloud_pod_run.sh <POD_NAME> <ZONE> [EXAMPLE_SCRIPT]
# Example: ./scripts/gcloud_pod_run.sh tpu-pod-v3-32 us-central1-a

set -euo pipefail

POD_NAME="${1:?Usage: $0 <POD_NAME> <ZONE> [SCRIPT]}"
ZONE="${2:?}"
SCRIPT="${3:-08_multi_host/train.py}"

# Resolve the coordinator IP (worker 0)
COORD_IP=$(gcloud compute tpus tpu-vm describe "$POD_NAME" \
  --zone="$ZONE" \
  --format="value(networkEndpoints[0].ipAddress)")

NUM_WORKERS=$(gcloud compute tpus tpu-vm describe "$POD_NAME" \
  --zone="$ZONE" \
  --format="value(networkEndpoints.len())")

echo "Pod: $POD_NAME  workers: $NUM_WORKERS  coordinator: $COORD_IP:8476"

# gcloud tpu-vm ssh with --worker=all runs the command on every host in parallel
gcloud compute tpus tpu-vm ssh "$POD_NAME" \
  --zone="$ZONE" \
  --worker=all \
  --command="
    WORKER_ID=\$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number' -H 'Metadata-Flavor: Google')
    python ~/tpu-examples/${SCRIPT} \
      --coordinator_address=${COORD_IP}:8476 \
      --num_processes=${NUM_WORKERS} \
      --process_id=\${WORKER_ID}
  "

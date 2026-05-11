#!/usr/bin/env bash
# Provision a preemptible Cloud TPU VM, install deps, and run hello-TPU.
# Defaults match DECISIONS.md: v5e-1 preemptible @ ~$0.36/hr in us-west4-a.
# Usage: ./scripts/provision_tpu.sh [TPU_NAME] [ZONE] [ACCELERATOR_TYPE]

set -euo pipefail

TPU_NAME="${1:-tpu-demo}"
ZONE="${2:-us-west4-a}"
ACCEL="${3:-v5litepod-1}"
RUNTIME="tpu-ubuntu2204-base"

echo "Creating preemptible TPU VM: $TPU_NAME ($ACCEL) in $ZONE ..."
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --zone="$ZONE" \
  --accelerator-type="$ACCEL" \
  --version="$RUNTIME" \
  --preemptible

echo "Copying repo ..."
gcloud compute tpus tpu-vm scp --recurse "$(pwd)" \
  "$TPU_NAME":~/tpu-examples --zone="$ZONE"

echo "Installing dependencies (jax[tpu] from Google libtpu index) ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="pip install --quiet -r ~/tpu-examples/requirements.txt"

echo "Installing OpenTelemetry Collector (otelcol-contrib v0.105.0) ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="
    set -e
    cd ~/tpu-examples
    OTELCOL_VERSION=0.105.0
    if [ ! -x ./otelcol-contrib ]; then
      curl -sLO \"https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v\${OTELCOL_VERSION}/otelcol-contrib_\${OTELCOL_VERSION}_linux_amd64.tar.gz\"
      tar -xzf otelcol-contrib_\${OTELCOL_VERSION}_linux_amd64.tar.gz otelcol-contrib
      chmod +x otelcol-contrib
      rm -f otelcol-contrib_\${OTELCOL_VERSION}_linux_amd64.tar.gz
    fi
    mkdir -p results/otel
  "

echo "Running hello-TPU check ..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --command="python ~/tpu-examples/01_hello_tpu/hello_tpu.py"

echo
echo "Done. Next steps:"
echo "  Smoke benchmark:  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \\"
echo "                      --command='cd tpu-examples && PYTHONPATH=. python benchmarks/harness.py --suite smoke --device tpu_v5e1'"
echo
echo "  OTel workflow (see DECISIONS.md ADR-014):"
echo "  1. Start OTel collector on TPU (separate SSH session):"
echo "       gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \\"
echo "         --command='cd tpu-examples && OUT_FILE=results/otel/\$(date +%s).jsonl ./otelcol-contrib --config infra/otelcol-tpu-config.yaml'"
echo "  2. Run benchmark with OTel enabled (in main SSH session):"
echo "       TPU_BENCH_OTEL=otlp PYTHONPATH=. python benchmarks/harness.py --suite smoke --device tpu_v5e1"
echo "  3. Stop the OTel collector (Ctrl+C in its SSH session)."
echo "  4. From your laptop:  ./scripts/otel_collect.sh && ./scripts/otel_view.sh"
echo
echo "  Teardown (stop billing!):  ./scripts/teardown_tpu.sh $TPU_NAME $ZONE"

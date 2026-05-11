#!/usr/bin/env bash
# Bring up the local Grafana + replay-otelcol stack and open the browser.
# See DECISIONS.md ADR-014 (local OTel + Grafana for TPU-run observability).
# Usage: ./scripts/otel_view.sh [--down]

set -euo pipefail

COMPOSE_FILE="infra/docker-compose.yml"
GRAFANA_URL="http://localhost:3000"

if [[ "${1:-}" == "--down" ]]; then
  echo "Stopping Grafana stack ..."
  docker compose -f "$COMPOSE_FILE" down
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' not found. Install Docker Desktop (or engine + compose plugin) first." >&2
  exit 1
fi

if [[ ! -d ./results/otel ]] || [[ -z "$(find ./results/otel -maxdepth 1 -name '*.jsonl' 2>/dev/null)" ]]; then
  echo "WARNING: ./results/otel/ is empty or missing — Grafana will start but show no data."
  echo "         Run ./scripts/otel_collect.sh first to pull traces from the TPU VM."
fi

echo "Starting Grafana stack (docker compose up -d) ..."
docker compose -f "$COMPOSE_FILE" up -d

echo "Waiting for Grafana to be healthy ..."
for i in $(seq 1 30); do
  if curl -sf "${GRAFANA_URL}/api/health" >/dev/null 2>&1; then
    echo "  Grafana is up after ${i}s."
    break
  fi
  if [[ $i -eq 30 ]]; then
    echo "  WARNING: Grafana did not report healthy within 30s; check 'docker compose logs grafana'."
  fi
  sleep 1
done

echo
echo "Grafana:    ${GRAFANA_URL}"
echo "Login:      admin / admin   (grafana/otel-lgtm default)"
echo
echo "Tail replay-collector ingest logs:"
echo "  docker compose -f ${COMPOSE_FILE} logs -f otel-replay"
echo
echo "Stop stack:  ./scripts/otel_view.sh --down"

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$GRAFANA_URL" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
  open "$GRAFANA_URL" >/dev/null 2>&1 &
elif command -v wslview >/dev/null 2>&1; then
  wslview "$GRAFANA_URL" >/dev/null 2>&1 &
else
  echo "(Open ${GRAFANA_URL} in your browser manually.)"
fi

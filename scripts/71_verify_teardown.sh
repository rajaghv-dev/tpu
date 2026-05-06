#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 71_verify_teardown.sh — Stage 7b: confirm zero billable resources remain.
#
# After 70_teardown_tpu.sh, run this to be SURE the burn rate is \$0/hr. It
# checks:
#   - any remaining TPU VMs in any zone we know about;
#   - running Compute Engine instances (in case a previous experiment left a
#     GPU VM lying around);
#   - reserved-but-idle static IPs;
#   - unattached persistent disks.
#
# Doesn't delete anything; only reports. Pair with 90_status.sh for ongoing
# monitoring during a session.
#
# Usage:
#   ./scripts/71_verify_teardown.sh
#
# Exit codes:
#   0 = no billable resources detected.
#   2 = at least one resource still costing money — investigate.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/config.sh"

STAGE="71_verify_teardown"
setup_error_trap
banner "Stage 7b — Verify zero billable resources"

require_cmd gcloud
remaining=0

# ── 1. TPUs ───────────────────────────────────────────────────────────────────
section "TPUs (across known zones)"
for z in $TPU_ZONES_PRIMARY $TPU_ZONES_FALLBACK; do
    rows=$(gcloud compute tpus tpu-vm list --zone="$z" \
        --format="value(name,state,acceleratorType)" 2>/dev/null | grep -v '^$' || true)
    if [[ -n "$rows" ]]; then
        log_warn "$z: still has TPU(s):"
        echo "$rows" | sed 's/^/    /' >&2
        remaining=$((remaining + $(echo "$rows" | wc -l)))
    fi
done
(( remaining == 0 )) && log_ok "no TPUs"

# ── 2. Compute Engine instances ───────────────────────────────────────────────
section "Compute Engine instances (any zone, RUNNING or TERMINATED)"
vms=$(gcloud compute instances list --format="value(name,zone,status,machineType.basename(),guestAccelerators[0].acceleratorType.basename())" 2>/dev/null || true)
if [[ -n "$vms" ]]; then
    log_warn "Found instances:"
    echo "$vms" | sed 's/^/    /' >&2
    remaining=$((remaining + $(echo "$vms" | wc -l)))
else
    log_ok "no instances"
fi

# ── 3. Reserved (unattached) static IPs ──────────────────────────────────────
# Static IPs that are RESERVED but not in use are billed at \$0.005/hr each.
section "static IPs"
ips=$(gcloud compute addresses list --filter="status=RESERVED" \
    --format="value(name,address,region.basename())" 2>/dev/null || true)
if [[ -n "$ips" ]]; then
    log_warn "Reserved-but-idle IPs:"
    echo "$ips" | sed 's/^/    /' >&2
    remaining=$((remaining + $(echo "$ips" | wc -l)))
else
    log_ok "no idle reserved IPs"
fi

# ── 4. Unattached persistent disks ────────────────────────────────────────────
# Persistent disks are billed even when no VM uses them (~\$0.04/GB-mo balanced).
section "unattached persistent disks"
disks=$(gcloud compute disks list --filter="-users:*" \
    --format="value(name,sizeGb,type,zone.basename())" 2>/dev/null || true)
if [[ -n "$disks" ]]; then
    log_warn "Disks with no users:"
    echo "$disks" | sed 's/^/    /' >&2
    remaining=$((remaining + $(echo "$disks" | wc -l)))
else
    log_ok "no orphan disks"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "summary"
if (( remaining == 0 )); then
    log_ok "Burn rate: \$0.00/hr — clean."
    exit 0
fi
log_warn "$remaining billable resource(s) remain. Investigate above."
log_warn "  Tip: ./scripts/90_status.sh estimates the hourly burn from these."
exit 2

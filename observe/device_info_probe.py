"""
observe/device_info_probe.py — one-shot SW/HW stack snapshot at run start.

Captures the full hardware and software stack the moment a benchmark begins:

  * host:      hostname, kernel, glibc, python, OS release, CPU model + topology,
               NUMA nodes, ISA feature flags, total RAM.
  * jax:       jax / jaxlib / libtpu versions, default backend, device list,
               process index/count.
  * tpu:       availability, chip count, device kind, topology, libtpu version,
               and a static HBM-per-chip / ICI-bandwidth lookup keyed on kind.
  * gpu:       nvidia-smi device list (if present), cuda runtime.
  * packages:  versions of the ML/numerics packages we depend on.
  * env_flags: snapshot of the JAX/XLA/TPU/NCCL env vars at start.

Why this probe matters
----------------------
Every result row in runs.jsonl is only useful if you can reproduce the
environment that produced it. The same JAX program can be 2x faster or 2x
slower depending on libtpu build, XLA_FLAGS, the TPU generation, or even
the host glibc that linked numpy. Attaching this snapshot to every run lets
us:

  * Explain why two timings on "the same" hardware differ.
  * Reproduce a run six months later by pinning every version this probe saw.
  * Diff stacks across hosts when a benchmark only fails on one of them.

It is a one-shot probe — values do not change during a run, so we only capture
in `before_run` and emit the dict from `write_log`.

Robustness
----------
Subprocess calls (nvidia-smi, nvcc) have a hard 3-second timeout so a hung
utility cannot stall a benchmark. Every optional dependency (jax, psutil) is
imported lazily inside a try/except, and every file read is guarded. On a
host with no jax, no psutil, and no nvidia-smi this probe still produces a
usable dict — just with most sub-sections marked `{"available": false, ...}`.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.probe import Probe

_log = logging.getLogger(__name__)

# Hard ceiling on every subprocess we shell out to. A hung nvidia-smi must
# never stall a benchmark — better to record "unavailable" than to block.
_SUBPROCESS_TIMEOUT_S: int = 3

# Static lookup for per-chip HBM and ICI bandwidth. Public spec sheet values;
# we record null if we don't recognise the device_kind so a stale table can't
# silently mislead a benchmark report.
_TPU_SPEC_TABLE: Dict[str, Dict[str, float]] = {
    "TPU v2":  {"hbm_gb": 8,  "ici_gbps": 100},
    "TPU v3":  {"hbm_gb": 16, "ici_gbps": 100},
    "TPU v4":  {"hbm_gb": 32, "ici_gbps": 270},
    "TPU v5e": {"hbm_gb": 16, "ici_gbps": 200},
    "TPU v5p": {"hbm_gb": 96, "ici_gbps": 600},
    "TPU v6e": {"hbm_gb": 32, "ici_gbps": 819},
}

# Subset of x86/ARM ISA flags we care about — affects whether numpy/XLA can
# use AVX-512 / AMX bf16 paths on the host side.
_ISA_FLAGS_OF_INTEREST: List[str] = [
    "avx2", "avx512f", "avx512bf16", "amx_bf16", "amx_tile", "sve", "neon",
]

_PACKAGES_OF_INTEREST: List[str] = [
    "jax", "jaxlib", "libtpu", "libtpu-nightly",
    "flax", "optax", "transformers",
    "numpy", "torch", "scipy",
    "orbax-checkpoint", "tensorboard", "wandb",
]

_ENV_VARS_OF_INTEREST: List[str] = [
    "JAX_PLATFORMS", "JAX_TRACEBACK_FILTERING", "JAX_ENABLE_X64",
    "JAX_DEBUG_NANS", "JAX_DISABLE_JIT",
    "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "TPU_ML_PLATFORM", "TPU_NAME", "TPU_WORKER_ID", "TPU_LOG_DIR",
    "LIBTPU_INIT_ARGS",
    "TF_CPP_MIN_LOG_LEVEL",
    "NCCL_DEBUG", "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
]


def _pkg_version(name: str) -> str:
    """Mirror lineage.get_package_version — same fallback string."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"
    except Exception:  # noqa: BLE001 — never raise from a probe
        return "unavailable"


def _run_cmd(argv: List[str]) -> Optional[str]:
    """Run argv with the global timeout. Return stdout or None on any failure."""
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except Exception:  # noqa: BLE001 — timeout, FileNotFound, perm error, …
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001 — missing file, perm error, /proc race
        return None


# ── host snapshot ─────────────────────────────────────────────────────────────

def _parse_os_release(text: str) -> Optional[str]:
    """Pull PRETTY_NAME (preferred) or NAME from /etc/os-release contents."""
    pretty: Optional[str] = None
    name: Optional[str] = None
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            pretty = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("NAME="):
            name = line.split("=", 1)[1].strip().strip('"')
    return pretty or name


def _cpu_model_and_flags(cpuinfo: str) -> Dict[str, Any]:
    """Extract first model-name line and the flag-set subset from /proc/cpuinfo."""
    model: Optional[str] = None
    flags_present: Dict[str, bool] = {f: False for f in _ISA_FLAGS_OF_INTEREST}
    for line in cpuinfo.splitlines():
        if model is None and (line.startswith("model name") or line.startswith("Model")):
            parts = line.split(":", 1)
            if len(parts) == 2:
                model = parts[1].strip()
        if line.startswith("flags") or line.startswith("Features"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                tokens = set(parts[1].split())
                for f in _ISA_FLAGS_OF_INTEREST:
                    if f in tokens:
                        flags_present[f] = True
            # don't break — some kernels print flags only for later CPUs
    return {"model": model, "isa_flags": flags_present}


def _numa_node_count() -> Optional[int]:
    """Count /sys/devices/system/node/node* dirs. None if sysfs unavailable."""
    try:
        base = Path("/sys/devices/system/node")
        if not base.is_dir():
            return None
        return sum(
            1 for entry in base.iterdir()
            if entry.is_dir() and re.fullmatch(r"node\d+", entry.name)
        )
    except Exception:  # noqa: BLE001
        return None


def _capture_host() -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": True}
    try:
        out["hostname"] = socket.gethostname()
    except Exception:  # noqa: BLE001
        out["hostname"] = "unavailable"

    try:
        out["platform"] = platform.platform()
    except Exception:  # noqa: BLE001
        out["platform"] = "unavailable"

    try:
        out["kernel"] = platform.uname().release
    except Exception:  # noqa: BLE001
        out["kernel"] = "unavailable"

    try:
        libc_name, libc_ver = platform.libc_ver()
        out["glibc"] = {"name": libc_name or None, "version": libc_ver or None}
    except Exception:  # noqa: BLE001
        out["glibc"] = None

    try:
        out["python_version"] = platform.python_version()
    except Exception:  # noqa: BLE001
        out["python_version"] = "unavailable"

    os_release_text = _read_text("/etc/os-release")
    out["os_release"] = _parse_os_release(os_release_text) if os_release_text else None

    cpuinfo = _read_text("/proc/cpuinfo")
    if cpuinfo:
        info = _cpu_model_and_flags(cpuinfo)
        out["cpu_model"] = info["model"]
        out["isa_flags"] = info["isa_flags"]
    else:
        out["cpu_model"] = None
        out["isa_flags"] = {f: None for f in _ISA_FLAGS_OF_INTEREST}

    # Core counts + RAM via psutil when available; fall back to os.cpu_count().
    physical_cores: Optional[int] = None
    logical_cores: Optional[int] = None
    total_ram_gb: Optional[float] = None
    try:
        import psutil  # type: ignore
        try:
            physical_cores = psutil.cpu_count(logical=False)
        except Exception:  # noqa: BLE001
            physical_cores = None
        try:
            logical_cores = psutil.cpu_count(logical=True)
        except Exception:  # noqa: BLE001
            logical_cores = None
        try:
            total_ram_gb = psutil.virtual_memory().total / (1024.0 ** 3)
        except Exception:  # noqa: BLE001
            total_ram_gb = None
    except Exception:  # noqa: BLE001 — psutil absent
        try:
            logical_cores = os.cpu_count()
        except Exception:  # noqa: BLE001
            logical_cores = None
    out["physical_cores"] = physical_cores
    out["logical_cores"] = logical_cores
    out["total_ram_gb"] = total_ram_gb

    out["numa_nodes"] = _numa_node_count()
    return out


# ── jax + tpu snapshot ────────────────────────────────────────────────────────

def _device_to_dict(dev: Any) -> Dict[str, Any]:
    """Best-effort projection of a jax Device — tolerate any AttributeError."""
    rec: Dict[str, Any] = {}
    for attr in ("id", "device_kind", "platform"):
        try:
            rec[attr] = getattr(dev, attr)
        except AttributeError:
            rec[attr] = None
        except Exception:  # noqa: BLE001
            rec[attr] = None
    # process_index is useful on multi-host pods; record if exposed.
    try:
        rec["process_index"] = getattr(dev, "process_index", None)
    except Exception:  # noqa: BLE001
        rec["process_index"] = None
    return rec


def _parse_tpu_topology_env() -> Optional[List[int]]:
    """Parse TPU_TOPOLOGY env var like '2x2' or '4x4x4' → list of ints."""
    raw = os.environ.get("TPU_TOPOLOGY")
    if not raw:
        return None
    try:
        parts = [int(p) for p in raw.lower().split("x")]
        return parts or None
    except Exception:  # noqa: BLE001
        return None


def _capture_jax_and_tpu() -> Dict[str, Dict[str, Any]]:
    """Returns (jax_section, tpu_section). jax is lazily imported."""
    libtpu_ver = _pkg_version("libtpu")
    if libtpu_ver == "not_installed":
        # Many TPU stacks ship nightly under a different dist name.
        nightly = _pkg_version("libtpu-nightly")
        if nightly != "not_installed":
            libtpu_ver = nightly

    try:
        import jax  # type: ignore
    except Exception as exc:  # noqa: BLE001 — jax is optional
        return {
            "jax": {"available": False, "reason": f"import failed: {exc!r}"},
            "tpu": {"available": False, "reason": "jax not importable"},
        }

    jax_section: Dict[str, Any] = {
        "available": True,
        "version": _pkg_version("jax"),
        "jaxlib_version": _pkg_version("jaxlib"),
        "libtpu_version": libtpu_ver,
    }

    try:
        jax_section["default_backend"] = jax.default_backend()
    except Exception as exc:  # noqa: BLE001
        jax_section["default_backend"] = None
        jax_section["default_backend_error"] = repr(exc)

    devices: List[Any] = []
    try:
        devices = list(jax.devices())
        jax_section["devices"] = [_device_to_dict(d) for d in devices]
    except Exception as exc:  # noqa: BLE001
        jax_section["devices"] = []
        jax_section["devices_error"] = repr(exc)

    for fn_name in ("process_index", "process_count", "local_device_count"):
        try:
            jax_section[fn_name] = getattr(jax, fn_name)()
        except Exception:  # noqa: BLE001
            jax_section[fn_name] = None

    # ── tpu sub-section ──
    backend = jax_section.get("default_backend")
    tpu_available = backend == "tpu"
    tpu_section: Dict[str, Any] = {"available": tpu_available}

    if not tpu_available:
        tpu_section["reason"] = (
            f"jax default backend is {backend!r}, not 'tpu'"
            if backend is not None
            else "jax.default_backend() unavailable"
        )
        tpu_section["libtpu_version"] = libtpu_ver
        return {"jax": jax_section, "tpu": tpu_section}

    device_kind: Optional[str] = None
    if devices:
        try:
            device_kind = getattr(devices[0], "device_kind", None)
        except Exception:  # noqa: BLE001
            device_kind = None

    chip_count: Optional[int] = jax_section.get("local_device_count")
    raw_count = len(devices) if devices else None
    topology_env = _parse_tpu_topology_env()

    tpu_section.update({
        "chip_count": chip_count,
        "device_kind": device_kind,
        "device_count_raw": raw_count,
        "topology_env": topology_env,
        "libtpu_version": libtpu_ver,
    })

    spec = _TPU_SPEC_TABLE.get(device_kind) if device_kind else None
    if spec is not None:
        tpu_section["hbm_per_chip_gb"] = spec["hbm_gb"]
        tpu_section["ici_bandwidth_gbps"] = spec["ici_gbps"]
    else:
        tpu_section["hbm_per_chip_gb"] = None
        tpu_section["ici_bandwidth_gbps"] = None

    return {"jax": jax_section, "tpu": tpu_section}


# ── gpu snapshot ──────────────────────────────────────────────────────────────

def _parse_nvidia_smi_csv(text: str) -> List[Dict[str, Any]]:
    """Parse the --format=csv,noheader output we requested into dicts."""
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        # We queried: index, name, memory.total, driver_version, compute_cap
        rec: Dict[str, Any] = {
            "index": cols[0] if len(cols) > 0 else None,
            "name": cols[1] if len(cols) > 1 else None,
            "memory_total": cols[2] if len(cols) > 2 else None,
            "driver_version": cols[3] if len(cols) > 3 else None,
            "compute_cap": cols[4] if len(cols) > 4 else None,
        }
        rows.append(rec)
    return rows


def _parse_nvcc_version(text: str) -> Optional[str]:
    """Pull the 'release X.Y' token from `nvcc --version` output."""
    m = re.search(r"release\s+([\d.]+)", text)
    return m.group(1) if m else None


def _capture_gpu() -> Dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not on PATH"}

    smi_out = _run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader",
    ])
    if smi_out is None:
        return {"available": False, "reason": "nvidia-smi present but call failed"}

    gpus = _parse_nvidia_smi_csv(smi_out)

    cuda_runtime: Optional[str] = None
    if shutil.which("nvcc") is not None:
        nvcc_out = _run_cmd(["nvcc", "--version"])
        if nvcc_out:
            cuda_runtime = _parse_nvcc_version(nvcc_out)

    return {
        "available": True,
        "gpus": gpus,
        "cuda_runtime": cuda_runtime,
    }


# ── packages + env ────────────────────────────────────────────────────────────

def _capture_packages() -> Dict[str, str]:
    return {name: _pkg_version(name) for name in _PACKAGES_OF_INTEREST}


def _capture_env_flags() -> Dict[str, Optional[str]]:
    return {name: os.environ.get(name) for name in _ENV_VARS_OF_INTEREST}


# ── probe ─────────────────────────────────────────────────────────────────────

class DeviceInfoProbe(Probe):
    """
    One-shot snapshot of host + accelerator + software stack at run start.

    Captures everything in `before_run`; emits a single nested dict from
    `write_log`. Every sub-section that fails to populate sets its own
    `{"available": false, "reason": ...}` rather than dropping keys, so
    downstream consumers can always count on the shape.
    """

    name = "device_info"

    def __init__(self) -> None:
        self._snapshot: Optional[Dict[str, Any]] = None

    def before_run(self, run_id: str, config: Any, log_dir: Path) -> None:
        snapshot: Dict[str, Any] = {
            "available": True,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            snapshot["host"] = _capture_host()
        except Exception as exc:  # noqa: BLE001
            _log.debug("device_info host section failed: %r", exc)
            snapshot["host"] = {"available": False, "reason": repr(exc)}

        try:
            jax_tpu = _capture_jax_and_tpu()
            snapshot["jax"] = jax_tpu["jax"]
            snapshot["tpu"] = jax_tpu["tpu"]
        except Exception as exc:  # noqa: BLE001 — must never raise
            _log.debug("device_info jax/tpu section failed: %r", exc)
            snapshot["jax"] = {"available": False, "reason": repr(exc)}
            snapshot["tpu"] = {"available": False, "reason": repr(exc)}

        try:
            snapshot["gpu"] = _capture_gpu()
        except Exception as exc:  # noqa: BLE001
            _log.debug("device_info gpu section failed: %r", exc)
            snapshot["gpu"] = {"available": False, "reason": repr(exc)}

        try:
            snapshot["packages"] = _capture_packages()
        except Exception as exc:  # noqa: BLE001
            _log.debug("device_info packages section failed: %r", exc)
            snapshot["packages"] = {"available": False, "reason": repr(exc)}

        try:
            snapshot["env_flags"] = _capture_env_flags()
        except Exception as exc:  # noqa: BLE001
            _log.debug("device_info env_flags section failed: %r", exc)
            snapshot["env_flags"] = {"available": False, "reason": repr(exc)}

        self._snapshot = snapshot

    def write_log(self) -> Optional[Dict[str, Any]]:
        if self._snapshot is None:
            # before_run never ran (e.g. probe registered after run start) —
            # still emit a usable shape so downstream tools don't trip.
            return {
                "available": False,
                "reason": "before_run was not invoked",
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
        return self._snapshot

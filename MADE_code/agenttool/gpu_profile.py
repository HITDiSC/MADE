"""Shared GPU profile resolution.

Both DockerSetUp and WeightResolve previously kept private copies of
``parse_use_gpu_config`` / ``query_gpu_inventory`` / ``resolve_gpu_profile``
that drifted apart over time. This module is the single source of truth.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agenttool.tool import linux_command


_FILE_DIR = os.path.dirname(__file__)
_PROJECT_DIR = os.path.dirname(_FILE_DIR)


# GPU architecture table. Each row maps a case-insensitive substring of the
# GPU product name (as reported by nvidia-smi) to an architecture class and
# the torch/CUDA constraints required to actually execute on that class.
#
# Fields:
#   arch           - coarse architecture code
#   sm             - compute capability (for LLM diagnostic context)
#   min_torch      - absolute minimum torch version with kernels for this arch
#                    (Blackwell needs >=2.6 etc.)
#   preferred_cuda - the cu-variant to prefer when the host allows it, chosen
#                    so torch official wheels are readily available
#   min_cuda       - absolute minimum CUDA toolkit the arch requires; going
#                    below this produces "no kernel image is available" at
#                    runtime even if torch itself installs cleanly
#
# Order matters: more specific patterns first (e.g. "rtx pro 6000 blackwell"
# must match before any shorter prefix).
_GPU_ARCH_TABLE: List[Tuple[str, Dict[str, str]]] = [
    # Blackwell (sm_100 / sm_120) - RTX 50-series, B100/B200, RTX PRO 6000 Blackwell
    ("rtx pro 6000 blackwell", {"arch": "blackwell", "sm": "sm_120", "min_torch": "2.6.0", "preferred_cuda": "12.4", "min_cuda": "12.4"}),
    ("rtx 50", {"arch": "blackwell", "sm": "sm_120", "min_torch": "2.6.0", "preferred_cuda": "12.4", "min_cuda": "12.4"}),
    ("b200", {"arch": "blackwell", "sm": "sm_100", "min_torch": "2.6.0", "preferred_cuda": "12.4", "min_cuda": "12.4"}),
    ("b100", {"arch": "blackwell", "sm": "sm_100", "min_torch": "2.6.0", "preferred_cuda": "12.4", "min_cuda": "12.4"}),
    # Hopper (sm_90) - H100/H200/H800
    ("h100", {"arch": "hopper", "sm": "sm_90", "min_torch": "2.0.0", "preferred_cuda": "12.1", "min_cuda": "11.8"}),
    ("h200", {"arch": "hopper", "sm": "sm_90", "min_torch": "2.0.0", "preferred_cuda": "12.1", "min_cuda": "11.8"}),
    ("h800", {"arch": "hopper", "sm": "sm_90", "min_torch": "2.0.0", "preferred_cuda": "12.1", "min_cuda": "11.8"}),
    # Ada Lovelace (sm_89) - RTX 40-series, L40/L40S/L4, RTX 6000 Ada
    ("rtx 6000 ada", {"arch": "ada_lovelace", "sm": "sm_89", "min_torch": "2.0.0", "preferred_cuda": "12.1", "min_cuda": "11.8"}),
    ("rtx 40", {"arch": "ada_lovelace", "sm": "sm_89", "min_torch": "2.0.0", "preferred_cuda": "12.1", "min_cuda": "11.8"}),
    ("l40s", {"arch": "ada_lovelace", "sm": "sm_89", "min_torch": "2.0.0", "preferred_cuda": "12.1", "min_cuda": "11.8"}),
    ("l40", {"arch": "ada_lovelace", "sm": "sm_89", "min_torch": "2.0.0", "preferred_cuda": "12.1", "min_cuda": "11.8"}),
    ("l4", {"arch": "ada_lovelace", "sm": "sm_89", "min_torch": "2.0.0", "preferred_cuda": "12.1", "min_cuda": "11.8"}),
    # Ampere (sm_80 / sm_86) - A100/A40/A30/A10, RTX 30-series, RTX A-series
    ("a100", {"arch": "ampere", "sm": "sm_80", "min_torch": "1.8.0", "preferred_cuda": "11.8", "min_cuda": "11.1"}),
    ("a40", {"arch": "ampere", "sm": "sm_86", "min_torch": "1.8.0", "preferred_cuda": "11.8", "min_cuda": "11.1"}),
    ("a30", {"arch": "ampere", "sm": "sm_80", "min_torch": "1.8.0", "preferred_cuda": "11.8", "min_cuda": "11.1"}),
    ("a10", {"arch": "ampere", "sm": "sm_86", "min_torch": "1.8.0", "preferred_cuda": "11.8", "min_cuda": "11.1"}),
    ("rtx a6000", {"arch": "ampere", "sm": "sm_86", "min_torch": "1.8.0", "preferred_cuda": "11.8", "min_cuda": "11.1"}),
    ("rtx a5000", {"arch": "ampere", "sm": "sm_86", "min_torch": "1.8.0", "preferred_cuda": "11.8", "min_cuda": "11.1"}),
    ("rtx a4000", {"arch": "ampere", "sm": "sm_86", "min_torch": "1.8.0", "preferred_cuda": "11.8", "min_cuda": "11.1"}),
    ("rtx 30", {"arch": "ampere", "sm": "sm_86", "min_torch": "1.8.0", "preferred_cuda": "11.8", "min_cuda": "11.1"}),
    # Turing (sm_75) - T4, RTX 20, Quadro RTX, GTX 16
    ("t4", {"arch": "turing", "sm": "sm_75", "min_torch": "1.4.0", "preferred_cuda": "11.8", "min_cuda": "10.0"}),
    ("rtx 20", {"arch": "turing", "sm": "sm_75", "min_torch": "1.4.0", "preferred_cuda": "11.8", "min_cuda": "10.0"}),
    ("quadro rtx", {"arch": "turing", "sm": "sm_75", "min_torch": "1.4.0", "preferred_cuda": "11.8", "min_cuda": "10.0"}),
    ("gtx 16", {"arch": "turing", "sm": "sm_75", "min_torch": "1.4.0", "preferred_cuda": "11.8", "min_cuda": "10.0"}),
    # Volta (sm_70) - V100
    ("v100", {"arch": "volta", "sm": "sm_70", "min_torch": "1.0.0", "preferred_cuda": "11.8", "min_cuda": "9.0"}),
    # Pascal (sm_60 / sm_61) - P100/P40/GTX 10
    ("p100", {"arch": "pascal", "sm": "sm_60", "min_torch": "1.0.0", "preferred_cuda": "11.8", "min_cuda": "9.0"}),
    ("p40", {"arch": "pascal", "sm": "sm_61", "min_torch": "1.0.0", "preferred_cuda": "11.8", "min_cuda": "9.0"}),
    ("gtx 10", {"arch": "pascal", "sm": "sm_61", "min_torch": "1.0.0", "preferred_cuda": "11.8", "min_cuda": "9.0"}),
]

# Newer architecture wins when multiple selected GPUs have different archs.
_ARCH_RANK: Dict[str, int] = {
    "pascal": 0, "volta": 1, "turing": 2, "ampere": 3,
    "ada_lovelace": 4, "hopper": 5, "blackwell": 6,
}


def _parse_version_tuple(v: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not v:
        return None
    try:
        return tuple(int(p) for p in str(v).split(".") if p.isdigit())
    except Exception:
        return None


def classify_gpu(name: Optional[str]) -> Optional[Dict[str, str]]:
    """Map a GPU product name (as reported by nvidia-smi) to its architecture
    constraints. Returns None when no entry matches - the caller MUST treat
    this as 'unknown' and fall back to driver_version / cuda_runtime_ceiling
    heuristics rather than guess.
    """
    if not name:
        return None
    lower = name.lower()
    for pattern, info in _GPU_ARCH_TABLE:
        if pattern in lower:
            return dict(info)
    return None


def compute_cuda_torch_hint(
    selected_gpus: List[Dict[str, Any]],
    cuda_runtime_ceiling: Optional[str],
) -> Dict[str, Any]:
    """Deterministic CUDA/torch recommendation for the selected GPUs.

    Output shape:
      {
        "resolved": bool,                # True only when at least one selected GPU matched the table
        "gpu_arch": str,                 # dominant arch, or "mixed:a,b" if heterogeneous
        "dominant_gpu_name": str,
        "sm": str,
        "min_torch_version": str,
        "recommended_cuda": str,         # e.g. "12.4"
        "recommended_torch_index_suffix": str,  # e.g. "cu124"
        "unknown_gpu_names": [str],      # GPUs that didn't match the table (may coexist with resolved=True)
        "conflict": bool,                # host ceiling < arch min_cuda (best-effort hint, will likely fail)
        "note": str,                     # human/LLM-readable explanation + escape-hatch guidance
      }

    When nothing resolves (no selected GPU, or none match the table), returns
    resolved=False with a note instructing the LLM to fall back to the
    cuda_runtime_ceiling / driver_version rules in the prompt. This is the
    deliberate escape hatch: the table will never cover every future NVIDIA
    SKU, so the LLM must still be able to act without a hint.
    """
    if not selected_gpus:
        return {
            "resolved": False,
            "note": (
                "No GPU selected. No CUDA/torch hint produced. Fall back to "
                "rules 11-14 using driver_version / cuda_runtime_ceiling, or "
                "omit CUDA-specific pins entirely if GPU is disabled."
            ),
        }

    classified: List[Tuple[str, Dict[str, str]]] = []
    unknown_names: List[str] = []
    for gpu in selected_gpus:
        name = gpu.get("name") or ""
        info = classify_gpu(name)
        if info:
            classified.append((name, info))
        else:
            if name:
                unknown_names.append(name)

    if not classified:
        return {
            "resolved": False,
            "unknown_gpu_names": unknown_names,
            "note": (
                f"Could not match any of {unknown_names or '[unnamed GPU]'} "
                "against the known GPU architecture table. Fall back to rules "
                "11-14: use driver_version / cuda_runtime_ceiling to pick a "
                "cu-variant and base image, and prefer a recent torch release."
            ),
        }

    # Dominant GPU = newest architecture among selected GPUs. A Blackwell
    # card in a mixed set raises the torch/CUDA floor for everyone else.
    classified.sort(key=lambda item: _ARCH_RANK.get(item[1]["arch"], -1), reverse=True)
    dominant_name, dominant = classified[0]

    preferred_cuda = dominant["preferred_cuda"]
    min_cuda = dominant["min_cuda"]
    recommended_cuda = preferred_cuda
    conflict = False

    ceiling_tuple = _parse_version_tuple(cuda_runtime_ceiling)
    preferred_tuple = _parse_version_tuple(preferred_cuda)
    min_tuple = _parse_version_tuple(min_cuda)

    if ceiling_tuple is not None:
        # If preferred CUDA exceeds the host driver ceiling, step down toward
        # the ceiling - but never below the arch's absolute minimum. When the
        # ceiling itself is below the minimum we flag a conflict and still
        # emit a best-effort recommendation so the LLM can at least produce
        # a Dockerfile (rebuild loop will surface the real failure).
        if preferred_tuple is not None and ceiling_tuple < preferred_tuple:
            if min_tuple is not None and ceiling_tuple < min_tuple:
                conflict = True
                recommended_cuda = min_cuda
            else:
                recommended_cuda = cuda_runtime_ceiling  # exact ceiling string

    cu_suffix = f"cu{recommended_cuda.replace('.', '')}" if recommended_cuda else None

    arches = sorted({info["arch"] for _, info in classified})
    arch_label = arches[0] if len(arches) == 1 else "mixed:" + ",".join(arches)

    if conflict:
        note = (
            f"CONFLICT: selected GPU(s) classified as {arch_label} "
            f"(dominant: {dominant_name}, {dominant['sm']}) require CUDA >= "
            f"{min_cuda}, but host cuda_runtime_ceiling is "
            f"{cuda_runtime_ceiling}. This combination cannot execute. "
            "Produce a best-effort Dockerfile using the recommended values "
            "below and surface the conflict in the 'reason' field."
        )
    else:
        note = (
            f"Selected GPU(s) classified as {arch_label} "
            f"(dominant: {dominant_name}, {dominant['sm']}). "
            f"torch must be >= {dominant['min_torch']}, built with CUDA "
            f">= {min_cuda}. Use wheel index "
            f"https://download.pytorch.org/whl/{cu_suffix}."
        )
        if unknown_names:
            note += (
                f" Note: {unknown_names} did not match the GPU table and were "
                "ignored when computing the hint."
            )

    return {
        "resolved": True,
        "gpu_arch": arch_label,
        "dominant_gpu_name": dominant_name,
        "sm": dominant["sm"],
        "min_torch_version": dominant["min_torch"],
        "recommended_cuda": recommended_cuda,
        "recommended_torch_index_suffix": cu_suffix,
        "unknown_gpu_names": unknown_names,
        "conflict": conflict,
        "note": note,
    }


def _load_global_config() -> Dict[str, Any]:
    config_path = os.path.join(_PROJECT_DIR, "config/global.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error in configuration file: {exc}")


GPU_DISABLED_MARKERS = {"", "none", "false", "no", "0"}


def parse_use_gpu_config(use_gpu: Any, inventory: List[Dict[str, Any]]) -> List[int]:
    """Resolve the user's ``use_gpu`` setting against the actual GPU inventory.

    Accepts:
        - bool (True -> all visible GPUs, False -> none)
        - "all" / "device=" / "device=all" -> all visible GPUs
        - "device=0,2" -> the listed indices
        - "2" -> first 2 visible GPUs
        - None / "none" / "false" / "no" / "0" / "" -> none
    """
    if isinstance(use_gpu, bool):
        return [item["index"] for item in inventory] if use_gpu else []

    use_gpu_str = "" if use_gpu is None else str(use_gpu).strip()
    use_gpu_str_lower = use_gpu_str.lower()
    if use_gpu_str_lower in GPU_DISABLED_MARKERS:
        return []

    if use_gpu_str_lower in {"all", "device=", "device=all"}:
        return [item["index"] for item in inventory]

    if use_gpu_str_lower.startswith("device="):
        device_spec = use_gpu_str.split("=", 1)[1].strip()
        if not device_spec:
            return [item["index"] for item in inventory]
        indices: List[int] = []
        for part in device_spec.split(","):
            part = part.strip()
            if part.isdigit():
                indices.append(int(part))
        return indices

    if use_gpu_str.isdigit():
        # "0" is already short-circuited via GPU_DISABLED_MARKERS above.
        requested_count = int(use_gpu_str)
        return [item["index"] for item in inventory[:requested_count]]

    return []


def query_gpu_inventory() -> List[Dict[str, Any]]:
    """Probe the local machine via nvidia-smi. Returns [] when nvidia-smi is missing."""
    try:
        result = linux_command(
            "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits"
        )
    except Exception:
        return []

    inventory: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index_str, name, memory_mb_str = parts[0], parts[1], parts[2]
        if not index_str.isdigit():
            continue
        try:
            memory_mb = int(float(memory_mb_str))
        except ValueError:
            memory_mb = None
        inventory.append({
            "index": int(index_str),
            "name": name,
            "memory_mb": memory_mb,
            "memory_gb": round(memory_mb / 1024, 2) if memory_mb is not None else None,
        })
    return inventory


def query_driver_version() -> Optional[str]:
    """Return the NVIDIA driver version string (e.g. '550.78') or None if unavailable."""
    try:
        result = linux_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    except Exception:
        return None
    for line in result.stdout.splitlines():
        v = line.strip()
        if v:
            return v
    return None


def query_cuda_runtime_ceiling() -> Optional[str]:
    """Return the maximum CUDA runtime version supported by the host driver (e.g. '12.4').

    This is what the ``nvidia-smi`` header reports as ``CUDA Version: X.Y``. It is the
    upper bound on the CUDA toolkit version that any container on this host can use,
    regardless of which CUDA toolkit ships inside the image. Returns None when
    nvidia-smi is missing or its output cannot be parsed.
    """
    try:
        result = linux_command("nvidia-smi")
    except Exception:
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
    return match.group(1) if match else None


def resolve_gpu_profile() -> Dict[str, Any]:
    """Build the canonical gpu_profile dict consumed by both DockerSetUp and WeightResolve."""
    global_config = _load_global_config()
    docker_setting = global_config.get("docker_setting") or {}
    use_gpu = docker_setting.get("use_gpu")

    inventory = query_gpu_inventory()
    selected_indices = parse_use_gpu_config(use_gpu, inventory)

    if isinstance(use_gpu, bool):
        gpu_enabled = use_gpu
    else:
        use_gpu_str = "" if use_gpu is None else str(use_gpu).strip().lower()
        gpu_enabled = use_gpu_str not in GPU_DISABLED_MARKERS

    inventory_by_index = {item["index"]: item for item in inventory}
    selected_gpus = [inventory_by_index[idx] for idx in selected_indices if idx in inventory_by_index]
    per_gpu_memory = [item["memory_gb"] for item in selected_gpus if item.get("memory_gb") is not None]

    return {
        "use_gpu": use_gpu,
        "gpu_enabled": gpu_enabled,
        "available_gpu_count": len(inventory),
        "selected_gpu_indices": selected_indices,
        "selected_gpus": selected_gpus,
        "gpu_count": len(selected_indices),
        "per_gpu_memory_gb": per_gpu_memory,
        "min_per_gpu_memory_gb": min(per_gpu_memory) if per_gpu_memory else None,
        "max_per_gpu_memory_gb": max(per_gpu_memory) if per_gpu_memory else None,
        "total_selected_memory_gb": round(sum(per_gpu_memory), 2) if per_gpu_memory else None,
        # Driver / CUDA compatibility ceiling. dockerfile_generation must respect
        # cuda_runtime_ceiling when picking a base image or torch wheel - the host
        # driver cannot run a CUDA toolkit higher than this value.
        "driver_version": query_driver_version(),
        "cuda_runtime_ceiling": query_cuda_runtime_ceiling(),
    }

"""File-based GPU lock for serializing inference across multiple containers.

All containers mount the same lock directory (e.g. -v /tmp/gpu_lock:/tmp/gpu_lock).
Only the container that holds the lock runs GPU inference; others block until
the lock is released. If a process crashes, the kernel auto-releases the flock.

Lock granularity: keyed by the ``use_gpu`` value from global.yaml (e.g.
``"device=0,1"``).  Tasks that share the same ``use_gpu`` config compete for
the same lock; tasks with different configs get independent locks and can run
in parallel on their respective GPUs.

Usage:
    from agenttool.gpu_lock import gpu_lock

    with gpu_lock():          # reads use_gpu from global.yaml automatically
        # start uvicorn, run inference, etc.
        ...
"""

import fcntl
import hashlib
import os
import re
from contextlib import contextmanager
from pathlib import Path

import yaml

# Lock file directory — must be on a shared mount across containers.
_LOCK_DIR = Path(os.environ.get("GPU_LOCK_DIR", "/tmp/gpu_lock"))


def _normalize_use_gpu(raw: str) -> str:
    """Canonicalize ``use_gpu`` so that ``"device=0,1"`` and ``"device=1,0"``
    resolve to the same lock.  ``"all"`` is kept as-is."""
    raw = raw.strip()
    if raw.lower() == "all":
        return "all"
    # Extract numeric indices, sort them, rejoin.
    indices = sorted(set(re.findall(r"\d+", raw)))
    if not indices:
        return raw
    return "device=" + ",".join(indices)


def _lock_file_for(use_gpu: str) -> Path:
    """Return the lock file path for a given (normalized) use_gpu value."""
    # Use a short hash to avoid filesystem-unfriendly characters.
    tag = hashlib.sha256(use_gpu.encode()).hexdigest()[:12]
    return _LOCK_DIR / f"gpu_{tag}.lock"


def _read_use_gpu() -> str:
    """Read ``docker_setting.use_gpu`` from global.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "global.yaml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return str((cfg.get("docker_setting") or {}).get("use_gpu", "all"))
    except Exception:
        return "all"


@contextmanager
def gpu_lock():
    """Acquire an exclusive file lock scoped to the current ``use_gpu`` config.

    Tasks with identical ``use_gpu`` block each other; tasks with different
    configs proceed independently.
    """
    use_gpu = _normalize_use_gpu(_read_use_gpu())
    lock_file = _lock_file_for(use_gpu)
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_file, "w")
    try:
        print(f"[gpu_lock] Waiting for GPU lock ({lock_file}, use_gpu={use_gpu}) ...")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        print(f"[gpu_lock] GPU lock acquired (pid={os.getpid()}, use_gpu={use_gpu})")
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        print(f"[gpu_lock] GPU lock released (pid={os.getpid()}, use_gpu={use_gpu})")

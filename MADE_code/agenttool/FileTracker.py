from __future__ import annotations

import json
import os
import hashlib
import difflib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from agenttool.tool import locate_path, build_tree

class FileTracker:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()
        self.meta_root = self.repo_root / ".autodeploy"

        self.manifest_path = self.meta_root / "manifest.json"
        self.current_state_path = self.meta_root / "current_state.json"

        self.file_records_dir = self.meta_root / "file_records"
        self.file_snapshots_dir = self.meta_root / "file_snapshots"
        self.file_diffs_dir = self.meta_root / "file_diffs"

    def init_workspace(self) -> None:
        self.meta_root.mkdir(exist_ok=True)
        self.file_records_dir.mkdir(exist_ok=True)
        self.file_snapshots_dir.mkdir(exist_ok=True)
        self.file_diffs_dir.mkdir(exist_ok=True)

        if not self.manifest_path.exists():
            self._write_json(self.manifest_path, {
                "workspace_root": str(self.repo_root),
                "created_at": self._now(),
                "tracked_files": []
            })

        if not self.current_state_path.exists():
            self._write_json(self.current_state_path, {})

    def modify_file(self, file_path: str, new_content: str) -> int:
        abs_path = self._abs_file_path(file_path)
        # Everything downstream (record / manifest / current_state / safe_key)
        # keys off a single canonical string form so lookups stay consistent
        # across Path vs str callers and survive a json.dump round-trip.
        abs_key = str(abs_path)
        old_content = self._read_text(abs_path)

        record = self._read_file_record(abs_key)
        current_version = record.get("current_version", 0)
        new_version = current_version + 1

        safe_key = self._safe_file_key(abs_key)
        snapshot_dir = self.file_snapshots_dir / safe_key
        diff_dir = self.file_diffs_dir / safe_key
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        diff_dir.mkdir(parents=True, exist_ok=True)

        before_snapshot = snapshot_dir / f"v{new_version:03d}_before"
        after_snapshot = snapshot_dir / f"v{new_version:03d}_after"
        diff_path = diff_dir / f"v{new_version:03d}.diff"

        self._write_text(before_snapshot, old_content)
        self._write_text(abs_path, new_content)
        self._write_text(after_snapshot, new_content)

        diff_text = "".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"{file_path} (before)",
                tofile=f"{file_path} (after)"
            )
        )
        self._write_text(diff_path, diff_text)

        history_item = {
            "version": new_version,
            "timestamp": self._now(),
            "before_snapshot": str(before_snapshot.relative_to(self.repo_root)),
            "after_snapshot": str(after_snapshot.relative_to(self.repo_root)),
            "diff_path": str(diff_path.relative_to(self.repo_root)),
            "hash_before": self._hash_text(old_content),
            "hash_after": self._hash_text(new_content)
        }

        if not record:
            record = {
                "file_path": abs_key,
                "current_version": 0,
                "history": []
            }

        record["current_version"] = new_version
        record["history"].append(history_item)
        self._write_file_record(abs_key, record)

        self._update_current_state(abs_key, new_version, history_item["after_snapshot"])
        self._update_manifest(abs_key)

        return new_version

    def rollback_file(self, file_path: str, version: Optional[int] = None) -> None:
        abs_path = self._abs_file_path(file_path)
        abs_key = str(abs_path)
        record = self._read_file_record(abs_key)
        if not record:
            raise FileNotFoundError(f"No history found for file: {file_path}")

        history = record.get("history", [])
        if not history:
            raise ValueError(f"No versions available for file: {file_path}")

        if version is None:
            if len(history) < 2:
                raise ValueError("No previous version to roll back to.")
            target_item = history[-2]
        else:
            matched = [item for item in history if item["version"] == version]
            if not matched:
                raise ValueError(f"Version {version} not found for file: {file_path}")
            target_item = matched[0]

        target_snapshot = self.repo_root / target_item["after_snapshot"]
        target_content = self._read_text(target_snapshot)

        self._write_text(abs_path, target_content)

        record["current_version"] = target_item["version"]
        self._write_file_record(abs_key, record)

        current_state = self._read_json(self.current_state_path)
        current_state[abs_key] = {
            "current_version": target_item["version"],
            "snapshot": target_item["after_snapshot"],
            "status": "tracked"
        }
        self._write_json(self.current_state_path, current_state)

    def get_file_history(self, file_path: str) -> List[Dict[str, Any]]:
        abs_path = self._abs_file_path(file_path)
        record = self._read_file_record(str(abs_path))
        return record.get("history", [])

    def get_current_version(self, file_path: str) -> Optional[int]:
        abs_path = self._abs_file_path(file_path)
        current_state = self._read_json(self.current_state_path)
        item = current_state.get(str(abs_path))
        if not item:
            return None
        return item.get("current_version")

    def _update_manifest(self, file_path: str) -> None:
        manifest = self._read_json(self.manifest_path)
        tracked_files = manifest.get("tracked_files", [])
        if file_path not in tracked_files:
            tracked_files.append(file_path)
        manifest["tracked_files"] = tracked_files
        manifest["updated_at"] = self._now()
        self._write_json(self.manifest_path, manifest)

    def _update_current_state(self, file_path: str, version: int, snapshot: str) -> None:
        current_state = self._read_json(self.current_state_path)
        current_state[file_path] = {
            "current_version": version,
            "snapshot": snapshot,
            "status": "tracked"
        }
        self._write_json(self.current_state_path, current_state)

    def _read_file_record(self, file_path: str) -> Dict[str, Any]:
        path = self.file_records_dir / f"{self._safe_file_key(file_path)}.json"
        return self._read_json(path)

    def _write_file_record(self, file_path: str, data: Dict[str, Any]) -> None:
        path = self.file_records_dir / f"{self._safe_file_key(file_path)}.json"
        self._write_json(path, data)

    def _abs_file_path(self, file_path: str) -> Path:
        found, current_path, node_type = locate_path(self.repo_root, file_path)
        if found:
            return Path(current_path).resolve()
        # locate_path only finds files that already exist on disk. For the
        # new-file case (adapt_code writing input_schema.json for the first
        # time, fix_runtime_code adding a new source file, etc.) accept any
        # absolute path: _read_text gracefully returns "" for non-existent
        # files, and _write_text creates parent dirs + the file itself.
        p = Path(file_path)
        if p.is_absolute():
            return p.resolve()
        raise FileNotFoundError(f"File not found: {file_path}")

    def _safe_file_key(self, file_path: str | Path) -> str:
        file_path_str = str(file_path)
        return file_path_str.replace("\\", "/").strip("/").replace("/", "__")

    def _now(self) -> str:
        return datetime.now().astimezone().isoformat()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_text(self, path: Path, content: str) -> None:
        self._atomic_write(path, content)

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # A corrupt tracker-metadata file (typically from a previous
            # non-atomic write that got killed mid-flush) would otherwise
            # wedge the whole pipeline on every subsequent modify_file call.
            # Self-heal by treating it as empty - the caller's next write
            # rebuilds the file cleanly via _atomic_write.
            print(f"[FileTracker] corrupt JSON at {path} ({e}); resetting to empty")
            return {}

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        self._atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))

    def _atomic_write(self, path: Path, content: str) -> None:
        """Write `content` to `path` such that any external observer sees
        either the old file or the new file, never a half-written one.

        Crashes / SIGKILL / disk-full during plain `open("w")` + stream-write
        leave the file truncated at whatever byte offset the kernel had
        flushed, which then fails json.load on the next read. Writing to a
        sibling tmp file + os.replace makes the final step a single rename
        (atomic on POSIX). Parent dir is created if missing so callers don't
        have to pre-mkdir.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            # Best-effort cleanup of the stray tmp on failure so repeated
            # retries don't leave a growing pile of *.tmp.<pid> files.
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise

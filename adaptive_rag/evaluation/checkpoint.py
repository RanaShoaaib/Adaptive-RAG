"""Durable JSONL checkpoints for evaluation runs."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

from adaptive_rag.evaluation.experiment import _write_json_atomic


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_record(path, record):
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("ab") as stream:
        stream.write(line.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def open_checkpoint(path: Path, config_path: Path, config: dict, *, id_key: str, legacy_paths=()):
    """Hold an exclusive writer lock while validating and using the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Checkpoint is already in use: {path}") from exc

        config = {"checkpoint_version": 1, **config}
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            if {k: v for k, v in saved.items() if k != "created_at_utc"} != {
                k: v for k, v in config.items() if k != "created_at_utc"
            }:
                raise ValueError(f"Checkpoint configuration mismatch: {config_path}. Use a new output.")
            config = saved
        else:
            if path.exists() or any(legacy.exists() for legacy in legacy_paths):
                raise ValueError(f"Checkpoint has no configuration: {path}. Use a new output.")
            _write_json_atomic(config_path, config)

        records = {}
        if path.exists():
            with path.open("r+b") as stream:
                line_number = 0
                while line := stream.readline():
                    line_number += 1
                    if not line.endswith(b"\n"):
                        # An interrupted append may leave an incomplete last record.
                        stream.truncate(stream.tell() - len(line))
                        break
                    try:
                        record = json.loads(line)
                        question_id = str(record[id_key])
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError(f"Invalid checkpoint record at {path}:{line_number}") from exc
                    if question_id in records:
                        raise ValueError(f"Duplicate question ID {question_id} in {path}")
                    records[question_id] = record
        yield config, records

"""Metadata-only DCP inspection, with no tensor payload reads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class TensorRecord:
    name: str
    kind: str
    shape: list[int] | None
    dtype: str | None
    chunks: list[dict[str, list[int]]]
    storage: list[dict[str, int | str | None]]


@dataclass(frozen=True)
class ShardRecord:
    path: str
    size_bytes: int | None
    references: int


@dataclass(frozen=True)
class CheckpointReport:
    checkpoint: str
    tensors: list[TensorRecord]
    shards: list[ShardRecord]
    findings: list[Finding]

    @property
    def ok(self) -> bool:
        return not any(finding.code.startswith("DCP1") for finding in self.findings)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "checkpoint": self.checkpoint,
            "format": "pytorch-dcp-filesystem",
            "summary": {
                "entries": len(self.tensors),
                "tensors": sum(record.kind == "tensor" for record in self.tensors),
                "non_tensor_entries": sum(record.kind != "tensor" for record in self.tensors),
                "shards": len(self.shards),
                "storage_file_bytes": sum(shard.size_bytes or 0 for shard in self.shards),
                "ok": self.ok,
            },
            "tensors": [asdict(record) for record in self.tensors],
            "shards": [asdict(shard) for shard in self.shards],
            "findings": [asdict(finding) for finding in self.findings],
        }


class InspectionError(Exception):
    """Raised when a checkpoint cannot be inspected."""


def _ints(value: Any) -> list[int]:
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise InspectionError("DCP metadata contains a non-integer shape") from exc


def _safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        return None
    return path.as_posix()


def _validate_shard(root: Path, relative: str) -> tuple[int | None, str | None]:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        stat = candidate.lstat()
    except FileNotFoundError:
        return None, "referenced storage file is missing"
    except OSError as exc:
        return None, f"cannot stat storage file ({exc.__class__.__name__})"
    if candidate.is_symlink():
        return None, "referenced storage file is a symlink"
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        return None, "referenced storage file escapes the checkpoint directory"
    if not candidate.is_file():
        return None, "referenced storage path is not a regular file"
    return stat.st_size, None


def _metadata_to_records(
    metadata: Any, root: Path
) -> tuple[list[TensorRecord], list[ShardRecord], list[Finding]]:
    state = getattr(metadata, "state_dict_metadata", None)
    storage = getattr(metadata, "storage_data", None)
    if not isinstance(state, dict):
        raise InspectionError("DCP metadata has no state_dict_metadata mapping")
    if not isinstance(storage, dict):
        raise InspectionError("DCP metadata has no filesystem storage_data mapping")

    storage_by_name: dict[str, list[dict[str, int | str | None]]] = {}
    shard_refs: dict[str, int] = {}
    shard_ranges: dict[str, list[tuple[int, int]]] = {}
    findings: list[Finding] = []

    for index, storage_info in storage.items():
        raw_name = getattr(storage_info, "relative_path", None)
        name = _safe_relative_path(raw_name)
        key_name = str(getattr(index, "fqn", "?"))
        if name is None:
            findings.append(
                Finding("DCP100", "unsafe or invalid storage path in metadata", key_name)
            )
            continue
        offset = getattr(storage_info, "offset", None)
        length = getattr(storage_info, "length", None)
        offset_int = int(offset) if isinstance(offset, int) else None
        length_int = int(length) if isinstance(length, int) else None
        storage_by_name.setdefault(key_name, []).append(
            {"path": name, "offset": offset_int, "length": length_int}
        )
        shard_refs[name] = shard_refs.get(name, 0) + 1
        if offset_int is not None and length_int is not None:
            shard_ranges.setdefault(name, []).append((offset_int, length_int))

    shard_records: list[ShardRecord] = []
    for name in sorted(shard_refs):
        size, issue = _validate_shard(root, name)
        shard_records.append(ShardRecord(name, size, shard_refs[name]))
        if issue:
            findings.append(Finding("DCP101", issue, name))
            continue
        for offset, length in shard_ranges.get(name, []):
            if offset < 0 or length < 0 or size is None or offset > size or length > size - offset:
                findings.append(
                    Finding("DCP102", "storage byte range exceeds shard file size", name)
                )
                break

    tensor_records: list[TensorRecord] = []
    for raw_name, value in sorted(state.items(), key=lambda item: str(item[0])):
        name = str(raw_name)
        raw_size = getattr(value, "size", None)
        properties = getattr(value, "properties", None)
        raw_chunks = getattr(value, "chunks", None)
        is_tensor = raw_size is not None and properties is not None and raw_chunks is not None
        if is_tensor:
            chunks = [
                {"offsets": _ints(chunk.offsets), "sizes": _ints(chunk.sizes)}
                for chunk in (raw_chunks or [])
            ]
            tensor_records.append(
                TensorRecord(
                    name=name,
                    kind="tensor",
                    shape=_ints(raw_size),
                    dtype=str(getattr(properties, "dtype", "unknown")),
                    chunks=chunks,
                    storage=sorted(
                        storage_by_name.get(name, []), key=lambda item: str(item["path"])
                    ),
                )
            )
        else:
            tensor_records.append(
                TensorRecord(
                    name=name,
                    kind=value.__class__.__name__,
                    shape=None,
                    dtype=None,
                    chunks=[],
                    storage=sorted(
                        storage_by_name.get(name, []), key=lambda item: str(item["path"])
                    ),
                )
            )
    return tensor_records, shard_records, findings


def inspect_checkpoint(checkpoint: str | Path) -> CheckpointReport:
    """Read DCP metadata and stat referenced shard files without reading tensor payloads."""
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise InspectionError(
            f"checkpoint directory does not exist or is not a directory: {checkpoint}"
        )
    metadata_path = root / ".metadata"
    try:
        metadata_stat = metadata_path.lstat()
    except FileNotFoundError as exc:
        raise InspectionError(f"DCP metadata file not found: {metadata_path}") from exc
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise InspectionError("DCP .metadata must be a regular, non-symlink file")
    if metadata_stat.st_size == 0:
        raise InspectionError("DCP .metadata file is empty")

    try:
        import torch.distributed.checkpoint as dcp
    except ImportError as exc:
        raise InspectionError(
            "PyTorch is required to read DCP metadata; install it in this environment "
            "(for example: pip install 'dcp-inspect[dcp]')"
        ) from exc

    try:
        reader_type = getattr(dcp, "FileSystemReader", None)
        if reader_type is None:
            raise InspectionError("installed PyTorch does not expose FileSystemReader")
        metadata = reader_type(str(root)).read_metadata()
    except Exception as exc:
        detail = str(exc).strip().replace("\n", " ")
        if len(detail) > 240:
            detail = detail[:237] + "..."
        suffix = f": {detail}" if detail else ""
        raise InspectionError(
            f"could not read DCP metadata ({exc.__class__.__name__}){suffix}"
        ) from exc

    tensors, shards, findings = _metadata_to_records(metadata, root)
    return CheckpointReport(root.name, tensors, shards, findings)

"""Compare DCP checkpoint schemas and storage layouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dcp_inspect.inspect import CheckpointReport, TensorRecord


@dataclass(frozen=True)
class Change:
    name: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None


@dataclass(frozen=True)
class CheckpointDiff:
    before: str
    after: str
    added: list[Change]
    removed: list[Change]
    schema_changed: list[Change]
    layout_changed: list[Change]
    unchanged: int

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.schema_changed or self.layout_changed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "before": self.before,
            "after": self.after,
            "summary": {
                "added": len(self.added),
                "removed": len(self.removed),
                "schema_changed": len(self.schema_changed),
                "layout_changed": len(self.layout_changed),
                "unchanged": self.unchanged,
                "has_changes": self.has_changes,
            },
            "added": [asdict(item) for item in self.added],
            "removed": [asdict(item) for item in self.removed],
            "schema_changed": [asdict(item) for item in self.schema_changed],
            "layout_changed": [asdict(item) for item in self.layout_changed],
        }


def _schema(record: TensorRecord) -> dict[str, Any]:
    return {"kind": record.kind, "shape": record.shape, "dtype": record.dtype}


def _layout(record: TensorRecord) -> dict[str, Any]:
    return {"chunks": record.chunks, "storage": record.storage}


def compare_checkpoints(before: CheckpointReport, after: CheckpointReport) -> CheckpointDiff:
    """Return added, removed, schema-changed, and layout-changed state entries."""
    left = {record.name: record for record in before.tensors}
    right = {record.name: record for record in after.tensors}
    added = [
        Change(name, None, _schema(right[name])) for name in sorted(right.keys() - left.keys())
    ]
    removed = [
        Change(name, _schema(left[name]), None) for name in sorted(left.keys() - right.keys())
    ]
    schema_changed: list[Change] = []
    layout_changed: list[Change] = []
    unchanged = 0
    for name in sorted(left.keys() & right.keys()):
        old, new = left[name], right[name]
        if _schema(old) != _schema(new):
            schema_changed.append(Change(name, _schema(old), _schema(new)))
        elif _layout(old) != _layout(new):
            layout_changed.append(Change(name, _layout(old), _layout(new)))
        else:
            unchanged += 1
    return CheckpointDiff(
        before.checkpoint,
        after.checkpoint,
        added,
        removed,
        schema_changed,
        layout_changed,
        unchanged,
    )

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from dcp_inspect.inspect import _metadata_to_records, _safe_relative_path


@dataclass(frozen=True)
class FakeIndex:
    fqn: str


def test_safe_relative_path_rejects_traversal_and_absolute_paths() -> None:
    assert _safe_relative_path("rank0/__0_0.distcp") == "rank0/__0_0.distcp"
    assert _safe_relative_path("../outside.distcp") is None
    assert _safe_relative_path("/outside.distcp") is None
    assert _safe_relative_path("rank0/../../outside.distcp") is None


def test_metadata_records_check_shard_ranges(tmp_path: Path) -> None:
    (tmp_path / "rank0.distcp").write_bytes(b"0123456789")
    state = {
        "weight": SimpleNamespace(
            size=(2, 2),
            properties=SimpleNamespace(dtype="float32"),
            chunks=[SimpleNamespace(offsets=(0, 0), sizes=(2, 2))],
        )
    }
    index = FakeIndex(fqn="weight")
    storage_info = SimpleNamespace(relative_path="rank0.distcp", offset=2, length=4)
    tensors, shards, findings = _metadata_to_records(
        SimpleNamespace(state_dict_metadata=state, storage_data={index: storage_info}), tmp_path
    )

    assert tensors[0].name == "weight"
    assert tensors[0].shape == [2, 2]
    assert tensors[0].dtype == "float32"
    assert shards[0].size_bytes == 10
    assert findings == []


def test_metadata_records_report_missing_files_and_invalid_ranges(tmp_path: Path) -> None:
    state = {}
    storage = {
        FakeIndex(fqn="missing"): SimpleNamespace(
            relative_path="missing.distcp", offset=0, length=1
        )
    }
    _, shards, findings = _metadata_to_records(
        SimpleNamespace(state_dict_metadata=state, storage_data=storage), tmp_path
    )

    assert shards[0].size_bytes is None
    assert [(finding.code, finding.path) for finding in findings] == [("DCP101", "missing.distcp")]


def test_metadata_records_reject_symlinked_shard(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.distcp"
    outside.write_bytes(b"data")
    (tmp_path / "linked.distcp").symlink_to(outside)
    storage = {
        FakeIndex(fqn="weight"): SimpleNamespace(relative_path="linked.distcp", offset=0, length=4)
    }

    _, _, findings = _metadata_to_records(
        SimpleNamespace(state_dict_metadata={}, storage_data=storage), tmp_path
    )

    assert [(finding.code, finding.path) for finding in findings] == [("DCP101", "linked.distcp")]


def test_metadata_records_reject_out_of_bounds_ranges(tmp_path: Path) -> None:
    (tmp_path / "rank0.distcp").write_bytes(b"0123456789")
    storage = {
        FakeIndex(fqn="weight"): SimpleNamespace(relative_path="rank0.distcp", offset=9, length=2)
    }

    _, _, findings = _metadata_to_records(
        SimpleNamespace(state_dict_metadata={}, storage_data=storage), tmp_path
    )

    assert [(finding.code, finding.path) for finding in findings] == [("DCP102", "rank0.distcp")]

from dcp_inspect.diff import compare_checkpoints
from dcp_inspect.inspect import CheckpointReport, ShardRecord, TensorRecord


def record(name: str, shape: list[int], *, shard: str = "rank0.distcp") -> TensorRecord:
    return TensorRecord(
        name=name,
        kind="tensor",
        shape=shape,
        dtype="torch.float32",
        chunks=[{"offsets": [0], "sizes": shape}],
        storage=[{"path": shard, "offset": 16, "length": 32}],
    )


def report(name: str, *records: TensorRecord) -> CheckpointReport:
    return CheckpointReport(name, list(records), [ShardRecord("rank0.distcp", 48, 1)], [])


def test_diff_classifies_keys_schema_and_layout() -> None:
    before = report("before", record("same", [4]), record("resized", [2]))
    after = report(
        "after",
        record("same", [4], shard="rank1.distcp"),
        record("resized", [3]),
        record("new", [1]),
    )

    result = compare_checkpoints(before, after)

    assert [item.name for item in result.added] == ["new"]
    assert [item.name for item in result.schema_changed] == ["resized"]
    assert [item.name for item in result.layout_changed] == ["same"]
    assert [item.name for item in result.removed] == []
    assert result.unchanged == 0
    assert result.as_dict()["summary"]["has_changes"] is True


def test_identical_checkpoints_have_no_changes() -> None:
    result = compare_checkpoints(
        report("a", record("weight", [4])), report("b", record("weight", [4]))
    )

    assert not result.has_changes
    assert result.unchanged == 1

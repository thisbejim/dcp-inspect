from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from dcp_inspect.cli import main
from dcp_inspect.diff import compare_checkpoints
from dcp_inspect.inspect import inspect_checkpoint

torch = pytest.importorskip("torch")
dcp = import_module("torch.distributed.checkpoint")


def save(directory: Path, width: int) -> None:
    dcp.save(
        {
            "model.proj.weight": torch.ones((2, width), dtype=torch.float32),
            "training.step": torch.tensor(1, dtype=torch.int64),
        },
        checkpoint_id=directory,
    )


def test_real_dcp_report_and_diff_without_distributed_workers(tmp_path: Path) -> None:
    before_path = tmp_path / "before"
    after_path = tmp_path / "after"
    save(before_path, 4)
    save(after_path, 8)

    before = inspect_checkpoint(before_path)
    after = inspect_checkpoint(after_path)
    difference = compare_checkpoints(before, after)

    assert before.ok
    assert after.ok
    assert before.as_dict()["summary"]["tensors"] == 2
    assert [change.name for change in difference.schema_changed] == ["model.proj.weight"]


def test_real_dcp_missing_shard_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    save(path, 4)
    shard = next(path.glob("*.distcp"))
    shard.unlink()

    report = inspect_checkpoint(path)

    assert not report.ok
    assert [(finding.code, finding.path) for finding in report.findings] == [("DCP101", shard.name)]


def test_missing_shard_report_fails_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "checkpoint"
    save(path, 4)
    shard = next(path.glob("*.distcp"))
    shard.unlink()

    assert main(["report", str(path)]) == 1
    assert "DCP101" in capsys.readouterr().out

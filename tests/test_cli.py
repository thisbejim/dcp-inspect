from __future__ import annotations

import pytest

from dcp_inspect.cli import main
from dcp_inspect.inspect import CheckpointReport, ShardRecord, TensorRecord


def fake_report(path: str) -> CheckpointReport:
    return CheckpointReport(
        path,
        [TensorRecord("weight", "tensor", [2], "torch.float32", [], [])],
        [ShardRecord("rank0.distcp", 8, 1)],
        [],
    )


def test_report_json_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("dcp_inspect.cli.inspect_checkpoint", fake_report)

    assert main(["report", "step-1", "--format", "json"]) == 0
    output = capsys.readouterr().out
    assert '"ok": true' in output
    assert '"name": "weight"' in output


def test_diff_fail_on_change(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("dcp_inspect.cli.inspect_checkpoint", fake_report)

    assert main(["diff", "step-1", "step-2", "--fail-on-change"]) == 0
    assert "0 schema changed" in capsys.readouterr().out

"""Command-line interface for DCP inspection and comparison."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from dcp_inspect.diff import CheckpointDiff, compare_checkpoints
from dcp_inspect.inspect import CheckpointReport, InspectionError, inspect_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dcp-inspect",
        description=(
            "Inspect and compare PyTorch Distributed Checkpoint metadata without loading tensors."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser(
        "report", help="summarize one checkpoint and validate its shard references"
    )
    report.add_argument("checkpoint", help="local DCP checkpoint directory")
    report.add_argument("--format", choices=("text", "json"), default="text")
    report.add_argument(
        "--max-entries", type=int, default=30, help="maximum entries shown in text mode"
    )
    diff = subparsers.add_parser("diff", help="compare tensor schemas and storage layouts")
    diff.add_argument("before", help="older/local DCP checkpoint directory")
    diff.add_argument("after", help="newer/local DCP checkpoint directory")
    diff.add_argument("--format", choices=("text", "json"), default="text")
    diff.add_argument(
        "--max-entries", type=int, default=30, help="maximum changes shown in text mode"
    )
    diff.add_argument(
        "--fail-on-change",
        action="store_true",
        help="exit 1 if any state or layout difference exists",
    )
    return parser


def _render_report(report: CheckpointReport, output_format: str, max_entries: int) -> str:
    if output_format == "json":
        return json.dumps(report.as_dict(), indent=2, sort_keys=True)
    summary = report.as_dict()["summary"]
    lines = [
        f"DCP checkpoint: {report.checkpoint}",
        f"Entries: {summary['entries']} ({summary['tensors']} tensors, "
        f"{summary['non_tensor_entries']} non-tensor)",
        f"Storage shards: {summary['shards']} | storage-file bytes: "
        f"{summary['storage_file_bytes']}",
        f"Status: {'PASS' if report.ok else 'FAIL'}",
    ]
    for finding in report.findings:
        location = f" ({finding.path})" if finding.path else ""
        lines.append(f"{finding.code}{location}: {finding.message}")
    if report.tensors:
        lines.append("Entries:")
        shown = report.tensors[:max_entries]
        for record in shown:
            if record.kind == "tensor":
                lines.append(
                    f"  {record.name}: {record.shape} {record.dtype} chunks={len(record.chunks)}"
                )
            else:
                lines.append(f"  {record.name}: {record.kind}")
        if len(report.tensors) > len(shown):
            lines.append(
                f"  … {len(report.tensors) - len(shown)} more; use --format json for the full list"
            )
    return "\n".join(lines)


def _render_diff(diff: CheckpointDiff, output_format: str, max_entries: int) -> str:
    if output_format == "json":
        return json.dumps(diff.as_dict(), indent=2, sort_keys=True)
    summary = diff.as_dict()["summary"]
    lines = [
        f"DCP diff: {diff.before} -> {diff.after}",
        "Entries: "
        f"+{summary['added']} added, -{summary['removed']} removed, "
        f"{summary['schema_changed']} schema changed, {summary['layout_changed']} layout changed, "
        f"{summary['unchanged']} unchanged",
    ]
    groups = (
        ("Added", diff.added),
        ("Removed", diff.removed),
        ("Schema changes", diff.schema_changed),
        ("Storage layout changes", diff.layout_changed),
    )
    count = 0
    for title, changes in groups:
        if not changes or count >= max_entries:
            continue
        lines.append(f"{title}:")
        for change in changes[: max_entries - count]:
            lines.append(f"  {change.name}: {change.before} -> {change.after}")
            count += 1
    total = (
        len(diff.added) + len(diff.removed) + len(diff.schema_changed) + len(diff.layout_changed)
    )
    if total > count:
        lines.append(f"  … {total - count} more changes; use --format json for the full list")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_entries < 0:
        parser.error("--max-entries must be zero or greater")
    try:
        if args.command == "report":
            report = inspect_checkpoint(args.checkpoint)
            print(_render_report(report, args.format, args.max_entries))
            return 0 if report.ok else 1
        before = inspect_checkpoint(args.before)
        after = inspect_checkpoint(args.after)
        diff = compare_checkpoints(before, after)
        print(_render_diff(diff, args.format, args.max_entries))
        if not before.ok or not after.ok:
            return 1
        return 1 if args.fail_on_change and diff.has_changes else 0
    except InspectionError as exc:
        print(f"dcp-inspect: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

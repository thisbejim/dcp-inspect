# Product specification

## Target developer

PyTorch research and infrastructure engineers who save or restore FSDP, DTensor, or other sharded state with filesystem-backed Distributed Checkpoint (DCP).

## Problem

Before attempting a distributed restore, an engineer may need to know whether the DCP state keys and tensor shapes still match expectations, whether a change is only a resharding/layout difference, and whether each referenced storage file is present. Today this often means constructing a model and launching the restore far enough to fail.

## Evidence

- PyTorch's DCP documentation explains that checkpoints consist of multiple files and loading is in-place into preallocated destination tensors.
- PyTorch issue #148252 demonstrates key compatibility behavior that differs from the familiar `load_state_dict(strict=False)` workflow.
- PyTorch issue #155924 reports state key name changes from compilation and activation checkpointing affecting save/load.
- TorchTitan's open checkpoint integrity RFC (#4011) describes DCP as a durable recovery point and calls out partial writes and incomplete checkpoint selection. That RFC proposes hashes; `dcp-inspect` addresses the separate metadata, schema, and file-reference preflight.

## Existing workflow and alternatives

- Calling `FileSystemReader.read_metadata()` directly is the supported low-level route, but requires a bespoke script to summarize and compare metadata.
- `dcp_to_torch_save` and restore-based checks convert or materialize tensor data, which is unnecessary for a schema/layout review.
- `weightlint` and `tensor-index-audit` audit Hugging Face-style bundles and indexes, rather than PyTorch's distributed `.metadata` and storage layout.

## Product thesis

For PyTorch distributed-training engineers, `dcp-inspect` makes DCP compatibility review quicker than a bespoke `FileSystemReader` script or a trial restore because it reports state schema changes and validates referenced shard files without loading tensors or initializing a process group.

## Core workflow

```text
DCP checkpoint directory (metadata + shard files)
    ↓
dcp-inspect report | diff
    ↓
Readable tensor schema, layout changes, and missing/invalid shard findings
```

## Non-goals

- Loading, converting, hashing, or authenticating tensor payloads.
- Proving optimizer or application-level resume compatibility.
- Supporting cloud/object-store or custom DCP storage readers.
- Constructing models, initializing distributed workers, or requiring GPUs.

## Interface and offline behavior

The primary interface is a local CLI with text and JSON reports. Once the package and PyTorch are installed, report and diff run offline. The package has no mandatory runtime dependencies; PyTorch is an optional extra and is used only to parse DCP metadata through its public filesystem reader.

## Privacy and security

No telemetry or network activity. The tool reads `.metadata`, stats shard files, and reports key/path metadata. Since PyTorch DCP metadata is serialized, users must trust checkpoint sources. Shard paths are checked for traversal and escaping symlinks before any file stat.

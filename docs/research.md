# Opportunity research

## Selected job

When a PyTorch training engineer is about to resume from a filesystem DCP checkpoint or compare a fresh checkpoint with its predecessor, they need to review the stored state schema and shard references without constructing the model and loading tensor payloads, so that avoidable compatibility and incomplete-file failures are found before a distributed job starts.

## Public demand signals

- [PyTorch issue #148252](https://github.com/pytorch/pytorch/issues/148252) documents a request for clearer/controllable handling of missing and unexpected DCP keys; the reporter shows a missing key fails inside the distributed load planning path.
- [PyTorch issue #155924](https://github.com/pytorch/pytorch/issues/155924) describes activation-checkpointing and compilation changing FQNs and breaking DCP save/load compatibility.
- [PyTorch DCP documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) says the checkpoint is multiple files with at least one per rank, loading is into preallocated tensors, and load-time resharding is supported. This makes pre-load metadata and file-reference inspection useful without reading model weights.
- [TorchTitan issue #4011](https://github.com/pytorch/torchtitan/issues/4011), opened in July 2026, proposes content integrity checks because a metadata marker alone can leave a partial checkpoint selected. This is adjacent evidence for the cost of validating checkpoints before resume, not evidence that this project implements content hashing.

Evidence is mostly upstream issue reports, official format behavior, and an open implementation proposal. It supports a concrete preflight job, but does not quantify how often engineers write this script themselves.

## Alternatives

| Alternative | Strength | Gap for this job |
| --- | --- | --- |
| `torch.distributed.checkpoint.FileSystemReader.read_metadata()` | Supported API exposes DCP metadata with no tensor restore. | Requires engineers to write and maintain their own report/diff and shard-reference checks. |
| PyTorch DCP restore or conversion APIs | Exercise the actual load path and can reshard. | Need a destination state, and may allocate/read large tensors or launch distributed workers just to discover metadata differences. |
| `weightlint` / `tensor-index-audit` | Useful offline bundle and shard-index checks. | Their supported inputs are Hugging Face weight bundles/indexes, not PyTorch DCP metadata and per-tensor chunk layout. |

## Candidate screen

1. **Draft/target tokenizer compatibility for speculative decoding.** Real upstream mismatch constraints, but it materially overlaps this workspace's `tokresolve` and the runtime's own compatibility checks.
2. **NCCL timeout log diagnosis.** Strong recurring pain signal, but PyTorch Flight Recorder and `torchfrtrace` already aggregate ranks and identify collective mismatch culprits.
3. **DCP schema and layout preflight.** Chosen. It targets a distinct input format and decision point, has low operational cost, and complements actual restore tests without reading tensor payloads.

## Skeptical review

Most DCP users can write a short Python script around `read_metadata()`, and some compatibility problems are best found by a real restore. This workflow is concentrated among PyTorch distributed-training teams, not every AI developer. The tool earns a narrow role by distinguishing state schema changes from shard-layout changes and checking file references in one reusable command; it does not claim to validate optimizer semantics or guarantee restore success. Its checks remain stable because they consume DCP's public metadata reader rather than parsing private binary formats.

The candidate passes the project threshold for a focused local utility, with the main caveat that the evidence establishes failure modes more clearly than frequency.

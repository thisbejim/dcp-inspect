# dcp-inspect

Inspect and compare PyTorch Distributed Checkpoint (DCP) metadata before a training job allocates a model or GPU memory.

DCP stores a checkpoint as metadata plus one or more tensor files. A load normally happens after the destination model has been constructed, and a key or shape mismatch can then fail inside distributed restore. `dcp-inspect` reads the small `.metadata` file, checks that its referenced local shard files and byte ranges exist, and reports tensor keys, shapes, dtypes, and storage layout. It does not load tensor payloads or start a process group.

## Install

PyTorch is optional at package install time and must be available when inspecting a DCP checkpoint:

```bash
python -m pip install "dcp-inspect[dcp] @ git+https://github.com/thisbejim/dcp-inspect.git@v0.1.0"
```

If PyTorch is already installed in the environment, install just the CLI:

```bash
python -m pip install "git+https://github.com/thisbejim/dcp-inspect.git@v0.1.0"
```

## Quick start

```console
$ dcp-inspect report checkpoints/step-12000
DCP checkpoint: step-12000
Entries: 3 (2 tensors, 1 non-tensor)
Storage shards: 1 | storage-file bytes: 4,415
Status: PASS
Entries:
  model.proj.bias: [2] torch.float32 chunks=1
  model.proj.weight: [2, 4] torch.float32 chunks=1
  training.step: BytesStorageMetadata
```

Compare two checkpoints before attempting a restore:

```bash
dcp-inspect diff checkpoints/step-11000 checkpoints/step-12000
```

Use a machine-readable report in scripts or CI:

```bash
dcp-inspect report checkpoints/step-12000 --format json > checkpoint.json
dcp-inspect diff checkpoints/before checkpoints/after --format json
```

`diff --fail-on-change` returns exit code 1 when keys, tensor schemas, or storage layouts differ. By default, differences are informational and return 0 if both checkpoints pass structural checks.

## What it checks

- Reads the filesystem DCP `.metadata` using `torch.distributed.checkpoint.FileSystemReader.read_metadata()`.
- Lists state entries; tensor shapes, dtypes, and chunk boundaries; and the storage file references recorded for each entry.
- Checks that referenced shards exist as regular files inside the checkpoint directory.
- Checks that recorded storage byte ranges are non-negative and fit within their shard files.
- Compares two checkpoints for added or removed state entries, shape/dtype changes, and shard-layout changes.
- Emits stable JSON with relative storage paths and no tensor values.

Tensor payload files are only stat'ed; their contents are never read. The tool does not need model code, a process group, a GPU, a network connection, or an API key. Filesystem checkpoints are its supported storage type; custom `StorageReader` implementations are outside scope.

## Interpreting a diff

- **Added / removed** means a state-dict entry exists in only one checkpoint.
- **Schema changed** means the entry's kind, global shape, or dtype changed.
- **Layout changed** means a tensor's global schema is the same, but its chunks or backing shard locations changed. This can happen when saving under a different distributed topology.

A clean schema diff does not prove that a model can load the checkpoint: optimizer semantics, application metadata, model code, and framework compatibility still matter. The command is a quick preflight, not a replacement for a real restore test.

## Security and privacy

All processing is local and read-only. There is no telemetry, account, or network access. Reports include state-dict names and relative shard paths; review them before sharing because names can reveal model architecture or training details.

PyTorch DCP metadata is a serialized PyTorch object. Only inspect checkpoints you trust, and use a patched PyTorch installation. `dcp-inspect` reads metadata through PyTorch's supported filesystem reader, does not deserialize tensor shard payloads, and rejects shard references that escape the checkpoint directory or resolve through symlinks.

## Development

```bash
python -m pip install -e ".[dev,dcp]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m build
```

The fixture tests create tiny CPU-only DCP checkpoints. They use no distributed workers, GPU, model download, or paid service.

## Why

PyTorch documents that DCP checkpoints contain multiple files and that loading allocates the destination tensors before DCP reads into them. Its `FileSystemReader.read_metadata()` API exposes useful metadata, but using it for a readable report or a before/after comparison otherwise takes a custom script. This CLI turns that metadata into a fast inspection step that can run before bringing up a distributed restore.

The project is narrower than checkpoint conversion tools: it never consolidates weights or materializes a tensor. It complements runtime restore tests by showing whether state keys and shard references changed.

## References

- [PyTorch Distributed Checkpoint documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch issue #148252: DCP missing and unexpected key behavior](https://github.com/pytorch/pytorch/issues/148252)
- [PyTorch issue #155924: compiled/activation-checkpointed FQNs and DCP](https://github.com/pytorch/pytorch/issues/155924)
- [TorchTitan issue #4011: checkpoint integrity verification proposal](https://github.com/pytorch/torchtitan/issues/4011)

## License

MIT. See [LICENSE](LICENSE).

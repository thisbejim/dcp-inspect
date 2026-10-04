"""Create a tiny CPU-only DCP directory for trying dcp-inspect."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--width", type=int, default=4)
    args = parser.parse_args()
    state = {
        "model.proj.weight": torch.arange(args.width * 2, dtype=torch.float32).reshape(
            2, args.width
        ),
        "model.proj.bias": torch.zeros(2, dtype=torch.float32),
        "training.step": 120,
    }
    dcp.save(state, checkpoint_id=args.directory)
    print(f"Wrote DCP checkpoint: {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

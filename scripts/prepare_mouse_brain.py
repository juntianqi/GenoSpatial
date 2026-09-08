#!/usr/bin/env python
"""Prepare and cache frozen GenoSpatial mouse-brain training inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genospatial.cache import prepare_and_cache
from genospatial.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an optional cache from authoritative GenoSpatial inputs."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_and_cache(config)
    print(
        f"Prepared {len(prepared.subcluster_names)} subclusters and "
        f"{len(prepared.expression_genes)} sequence-matched genes; "
        f"cache written to {config['cache']['path']}"
    )


if __name__ == "__main__":
    main()


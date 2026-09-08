#!/usr/bin/env python
"""Train frozen GenoSpatial main model from authoritative upstream inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genospatial.cache import prepare_or_load
from genospatial.config import load_config
from genospatial.dataset import build_dataloaders
from genospatial.training import build_model, resolve_device, train
from genospatial.utils import write_checkpoint_metadata, write_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen GenoSpatial mouse-brain main model."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_or_load(config)
    train_loader, validation_loader, _ = build_dataloaders(prepared, config)
    model = build_model(prepared, config)
    device = resolve_device(config["training"]["device"])
    checkpoint_path = (
        Path(config["output"]["directory"])
        / config["output"]["checkpoint_name"]
    )
    write_checkpoint_metadata(checkpoint_path, config, prepared)
    history = train(model, train_loader, validation_loader, config, device)
    history_path = Path(config["output"]["directory"]) / config["output"]["history_name"]
    write_training_history(history_path, history)
    print(
        f"Training finished after {len(history.validation_losses)} "
        f"validation events; checkpoint target: {checkpoint_path}; "
        f"history: {history_path}"
    )


if __name__ == "__main__":
    main()

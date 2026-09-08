"""YAML configuration loading and validation for GenoSpatial training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


VALID_CACHE_MODES = {"off", "auto", "require"}
REQUIRED_SECTIONS = {
    "inputs",
    "data",
    "split",
    "model",
    "training",
    "loss",
    "optimizer",
    "early_stopping",
    "output",
    "cache",
}
REQUIRED_INPUTS = (
    "expression_h5ad",
    "stcase_h5ad",
    "sequence_hdf5",
    "gtf",
    "cell_type_csv",
)


class ConfigError(ValueError):
    """Raised when a training configuration is missing or inconsistent."""


def _require_keys(mapping: Mapping[str, Any], keys, context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"Missing required {context}: {', '.join(missing)}")


def validate_config(config: Mapping[str, Any], *, check_input_files: bool = True) -> None:
    """Validate required structure without changing configured values."""

    if not isinstance(config, Mapping):
        raise ConfigError("Configuration root must be a mapping")
    _require_keys(config, REQUIRED_SECTIONS, "configuration sections")
    _require_keys(config["inputs"], REQUIRED_INPUTS, "inputs")

    cache_mode = config["cache"].get("mode")
    if cache_mode not in VALID_CACHE_MODES:
        raise ConfigError(
            f"cache.mode must be one of {sorted(VALID_CACHE_MODES)}, got {cache_mode!r}"
        )

    split = config["split"]
    _require_keys(
        split,
        {"train_chromosomes", "validation_chromosomes", "test_chromosomes"},
        "chromosome split fields",
    )
    split_sets = [
        set(split["train_chromosomes"]),
        set(split["validation_chromosomes"]),
        set(split["test_chromosomes"]),
    ]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ConfigError("Chromosome split lists must be disjoint")

    for key in ("minimum_niche_cells", "minimum_subcluster_cells"):
        value = config["data"].get(key)
        if not isinstance(value, int) or value < 1:
            raise ConfigError(f"data.{key} must be a positive integer")

    for key in ("seq_len", "seq_channels", "cell_embed_dim", "environment_dim", "hidden_dim", "num_heads", "num_factors"):
        value = config["model"].get(key)
        if not isinstance(value, int) or value < 1:
            raise ConfigError(f"model.{key} must be a positive integer")
    if config["model"]["hidden_dim"] % config["model"]["num_heads"] != 0:
        raise ConfigError("model.hidden_dim must be divisible by model.num_heads")

    for key in ("branch_dropout_probability", "predictor_dropout_probability"):
        value = config["model"].get(key)
        if not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ConfigError(f"model.{key} must be in [0, 1)")

    for key in ("epochs", "batch_size", "validation_batch_size", "test_batch_size", "validation_every_steps"):
        value = config["training"].get(key)
        if not isinstance(value, int) or value < 1:
            raise ConfigError(f"training.{key} must be a positive integer")
    for key in ("train_workers", "validation_workers", "test_workers"):
        value = config["training"].get(key)
        if not isinstance(value, int) or value < 0:
            raise ConfigError(f"training.{key} must be a non-negative integer")

    if config["loss"].get("name") != "HuberLoss":
        raise ConfigError("loss.name must be HuberLoss for the frozen implementation")
    if not isinstance(config["loss"].get("delta"), (int, float)) or config["loss"]["delta"] <= 0:
        raise ConfigError("loss.delta must be positive")
    if config["optimizer"].get("name") != "AdamW":
        raise ConfigError("optimizer.name must be AdamW for the frozen implementation")

    if check_input_files:
        for key in REQUIRED_INPUTS:
            path = Path(config["inputs"][key]).expanduser()
            if not path.is_file():
                raise ConfigError(f"inputs.{key} does not exist or is not a file: {path}")


def load_config(path: str | Path, *, check_input_files: bool = True) -> dict[str, Any]:
    """Load a YAML configuration and validate it."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config, check_input_files=check_input_files)
    return dict(config)

"""Operational metadata helpers for the public training interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_checkpoint_metadata(
    output_path: str | Path, config: Mapping[str, Any], prepared
) -> Path:
    """Write semantic ordering metadata alongside the state-dict checkpoint."""

    checkpoint_path = Path(output_path)
    metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": checkpoint_path.name,
        "authoritative_random_seed": None,
        "random_seed_status": "UNRESOLVED: no explicit seed in authoritative notebook",
        "cell_type_to_id": prepared.cell_type_to_id,
        "lr_feature_names": prepared.lr_feature_names,
        "expression_genes": prepared.expression_genes,
        "train_genes": prepared.train_genes,
        "validation_genes": prepared.validation_genes,
        "test_genes": prepared.test_genes,
        "subcluster_names": prepared.subcluster_names,
        "config": _json_value(config),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata_path


def write_training_history(output_path: str | Path, history) -> Path:
    """Write validation-event history produced by the frozen training loop."""

    history_path = Path(output_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_steps": history.global_steps,
        "cumulative_epoch_train_weighted_huber": history.train_losses,
        "validation_weighted_huber": history.validation_losses,
        "validation_pearson_r": history.validation_correlations,
    }
    history_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return history_path

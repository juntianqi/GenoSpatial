from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from genospatial.config import ConfigError, load_config, validate_config


def _base_config(tmp_path: Path) -> dict:
    return {
        "inputs": {
            "expression_h5ad": str(tmp_path / "expression.h5ad"),
            "stcase_h5ad": str(tmp_path / "stcase.h5ad"),
            "sequence_hdf5": str(tmp_path / "sequence.h5"),
            "gtf": str(tmp_path / "genes.gtf"),
            "cell_type_csv": str(tmp_path / "cell_types.csv"),
        },
        "data": {
            "cell_type_column": "cell_type_sub",
            "spatial_cluster_column": "X_hg_final_cluster",
            "environment_key": "X_hg_final_emb",
            "ccc_sender_key": "X_ccc_sender",
            "ccc_receiver_key": "X_ccc_receiver",
            "subcluster_column": "spatial_sub_cluster",
            "lr_information_key": "LR_pair_information",
            "lr_complex_key": "DB_complex",
            "lr_weight_key": "LR_cell_weight",
            "minimum_niche_cells": 500,
            "minimum_subcluster_cells": 150,
        },
        "split": {
            "train_chromosomes": [f"chr{i}" for i in range(1, 13)] + ["chrX"],
            "validation_chromosomes": ["chr13", "chr14", "chr15"],
            "test_chromosomes": ["chr16", "chr17", "chr18", "chr19"],
        },
        "model": {
            "seq_len": 624,
            "seq_channels": 1920,
            "cell_embed_dim": 128,
            "environment_dim": 128,
            "hidden_dim": 128,
            "num_heads": 4,
            "num_factors": 8,
            "branch_dropout_probability": 0.25,
            "predictor_dropout_probability": 0.2,
        },
        "training": {
            "epochs": 150,
            "batch_size": 256,
            "validation_batch_size": 256,
            "test_batch_size": 128,
            "train_workers": 4,
            "validation_workers": 0,
            "test_workers": 0,
            "validation_every_steps": 250,
            "device": "auto",
        },
        "loss": {"name": "HuberLoss", "delta": 0.1},
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 5e-5,
            "weight_decay": 1e-3,
        },
        "early_stopping": {"patience": 10, "minimum_delta": 1e-4},
        "output": {
            "directory": str(tmp_path / "outputs"),
            "checkpoint_name": "best_Advtrimodal_model_v12_r.pth",
            "history_name": "training_history.json",
        },
        "cache": {"mode": "off", "path": str(tmp_path / "prepared.h5")},
    }


def test_load_config_preserves_frozen_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_base_config(tmp_path)), encoding="utf-8")

    config = load_config(config_path, check_input_files=False)

    assert config["model"]["seq_len"] == 624
    assert config["model"]["seq_channels"] == 1920
    assert config["model"]["num_factors"] == 8
    assert config["training"]["batch_size"] == 256
    assert config["training"]["validation_every_steps"] == 250
    assert config["optimizer"] == {
        "name": "AdamW",
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
    }


def test_actual_distributed_configs_parse_modes_and_preserve_frozen_defaults():
    direct = load_config(
        PROJECT_ROOT / "configs" / "mouse_brain.yaml",
        check_input_files=False,
    )
    cached = load_config(
        PROJECT_ROOT / "configs" / "mouse_brain_cached.yaml",
        check_input_files=False,
    )

    assert direct["cache"]["mode"] == "off"
    assert cached["cache"]["mode"] == "require"
    for config in (direct, cached):
        assert config["data"]["minimum_niche_cells"] == 500
        assert config["data"]["minimum_subcluster_cells"] == 150
        assert config["split"] == {
            "train_chromosomes": [f"chr{i}" for i in range(1, 13)] + ["chrX"],
            "validation_chromosomes": ["chr13", "chr14", "chr15"],
            "test_chromosomes": ["chr16", "chr17", "chr18", "chr19"],
        }
        assert config["model"] == {
            "seq_len": 624,
            "seq_channels": 1920,
            "cell_embed_dim": 128,
            "environment_dim": 128,
            "hidden_dim": 128,
            "num_heads": 4,
            "num_factors": 8,
            "branch_dropout_probability": 0.25,
            "predictor_dropout_probability": 0.2,
        }
        assert config["training"] == {
            "epochs": 150,
            "batch_size": 256,
            "validation_batch_size": 256,
            "test_batch_size": 128,
            "train_workers": 4,
            "validation_workers": 0,
            "test_workers": 0,
            "validation_every_steps": 250,
            "device": "auto",
        }
        assert config["loss"] == {"name": "HuberLoss", "delta": 0.1}
        assert config["optimizer"] == {
            "name": "AdamW",
            "learning_rate": 5e-5,
            "weight_decay": 1e-3,
        }
        assert config["early_stopping"] == {
            "patience": 10,
            "minimum_delta": 1e-4,
        }


def test_validate_config_rejects_missing_section(tmp_path):
    config = _base_config(tmp_path)
    del config["split"]
    with pytest.raises(ConfigError, match="split"):
        validate_config(config, check_input_files=False)


def test_validate_config_rejects_invalid_cache_mode(tmp_path):
    config = _base_config(tmp_path)
    config["cache"]["mode"] = "sometimes"
    with pytest.raises(ConfigError, match="cache.mode"):
        validate_config(config, check_input_files=False)


def test_validate_config_rejects_missing_input_file(tmp_path):
    with pytest.raises(ConfigError, match="expression_h5ad"):
        validate_config(_base_config(tmp_path), check_input_files=True)


def test_validate_config_allows_test_only_threshold_overrides(tmp_path):
    config = _base_config(tmp_path)
    config["data"]["minimum_niche_cells"] = 2
    config["data"]["minimum_subcluster_cells"] = 1
    validate_config(config, check_input_files=False)

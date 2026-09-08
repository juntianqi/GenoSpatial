"""Optional, validated cache for authoritative preprocessing outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .preprocessing import PreparedTrainingData, prepare_training_data


CACHE_SCHEMA_VERSION = 1


class CacheError(RuntimeError):
    """Raised when a preprocessing cache is missing, stale or malformed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_cache_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a lightweight signature from inputs and preprocessing settings."""

    input_identity: dict[str, dict[str, Any]] = {}
    for name, value in config["inputs"].items():
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise CacheError(f"Cannot fingerprint missing input: inputs.{name}")
        stat = path.stat()
        input_identity[name] = {
            "path": str(path),
            "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }
    model_keys = ("seq_len", "seq_channels", "environment_dim")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "inputs": input_identity,
        "data": dict(config["data"]),
        "split": dict(config["split"]),
        "model_input_contract": {
            key: config["model"][key] for key in model_keys
        },
    }


def _write_strings(handle: h5py.File, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def _read_strings(handle: h5py.File, name: str) -> list[str]:
    values = handle[name].asstr()[:]
    return [str(value) for value in values.tolist()]


def save_prepared_cache(
    path: str | Path,
    prepared: PreparedTrainingData,
    signature: Mapping[str, Any],
) -> None:
    """Write a complete preprocessing result and its validation signature."""

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["signature_json"] = _canonical_json(signature)
        handle.attrs["gene_weights_json"] = _canonical_json(prepared.gene_weights)
        handle.attrs["cell_type_to_id_json"] = _canonical_json(prepared.cell_type_to_id)
        handle.attrs["sequence_hdf5_path"] = prepared.sequence_hdf5_path
        handle.create_dataset("major_ids", data=prepared.major_ids)
        handle.create_dataset("mean_environment", data=prepared.mean_environment)
        handle.create_dataset("mean_receiver_ccc", data=prepared.mean_receiver_ccc)
        handle.create_dataset("mean_expression", data=prepared.mean_expression)
        _write_strings(handle, "subcluster_names", prepared.subcluster_names)
        _write_strings(handle, "expression_genes", prepared.expression_genes)
        _write_strings(handle, "train_genes", prepared.train_genes)
        _write_strings(handle, "validation_genes", prepared.validation_genes)
        _write_strings(handle, "test_genes", prepared.test_genes)
        _write_strings(handle, "lr_feature_names", prepared.lr_feature_names)


def load_prepared_cache(
    path: str | Path,
    *,
    expected_signature: Mapping[str, Any] | None = None,
) -> PreparedTrainingData:
    """Load a cache and reject it when its signature does not match."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise CacheError(f"Prepared-data cache does not exist: {cache_path}")
    try:
        with h5py.File(cache_path, "r") as handle:
            stored_signature = json.loads(handle.attrs["signature_json"])
            if expected_signature is not None and _canonical_json(stored_signature) != _canonical_json(expected_signature):
                raise CacheError("Prepared-data cache signature does not match current inputs/config")
            return PreparedTrainingData(
                subcluster_names=_read_strings(handle, "subcluster_names"),
                major_ids=handle["major_ids"][:],
                mean_environment=handle["mean_environment"][:],
                mean_receiver_ccc=handle["mean_receiver_ccc"][:],
                mean_expression=handle["mean_expression"][:],
                expression_genes=_read_strings(handle, "expression_genes"),
                train_genes=_read_strings(handle, "train_genes"),
                validation_genes=_read_strings(handle, "validation_genes"),
                test_genes=_read_strings(handle, "test_genes"),
                gene_weights={
                    str(key): float(value)
                    for key, value in json.loads(handle.attrs["gene_weights_json"]).items()
                },
                cell_type_to_id={
                    str(key): int(value)
                    for key, value in json.loads(handle.attrs["cell_type_to_id_json"]).items()
                },
                lr_feature_names=_read_strings(handle, "lr_feature_names"),
                sequence_hdf5_path=str(handle.attrs["sequence_hdf5_path"]),
            )
    except CacheError:
        raise
    except Exception as exc:
        raise CacheError(f"Prepared-data cache is malformed: {cache_path}") from exc


def prepare_or_load(config: Mapping[str, Any]) -> PreparedTrainingData:
    """Follow configured cache semantics while retaining one preparation path."""

    mode = config["cache"]["mode"]
    cache_path = Path(config["cache"]["path"])
    if mode == "off":
        return prepare_training_data(config)

    signature = build_cache_signature(config)
    if mode == "require":
        return load_prepared_cache(cache_path, expected_signature=signature)
    if mode == "auto" and cache_path.is_file():
        try:
            return load_prepared_cache(cache_path, expected_signature=signature)
        except CacheError:
            pass
    prepared = prepare_training_data(config)
    save_prepared_cache(cache_path, prepared, signature)
    return prepared


def prepare_and_cache(config: Mapping[str, Any]) -> PreparedTrainingData:
    """Recompute authoritative preprocessing and write the configured cache."""

    prepared = prepare_training_data(config)
    signature = build_cache_signature(config)
    save_prepared_cache(config["cache"]["path"], prepared, signature)
    return prepared

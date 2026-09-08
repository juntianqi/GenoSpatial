from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genospatial.cache import CacheError, load_prepared_cache, save_prepared_cache
from genospatial.preprocessing import PreparedTrainingData


def _prepared(tmp_path: Path) -> PreparedTrainingData:
    return PreparedTrainingData(
        subcluster_names=["A_n1", "B_n2"],
        major_ids=np.array([0, 1], dtype=np.int64),
        mean_environment=np.array([[1.0, 2.0], [3.0, 4.0]]),
        mean_receiver_ccc=np.array([[5.0], [6.0]]),
        mean_expression=np.array([[0.1, 0.2], [0.3, 0.4]]),
        expression_genes=["g1", "g2"],
        train_genes=["g1"],
        validation_genes=["g2"],
        test_genes=[],
        gene_weights={"g1": 1.5, "g2": 2.0},
        cell_type_to_id={"A": 0, "B": 1},
        lr_feature_names=["lr_a"],
        sequence_hdf5_path=str(tmp_path / "sequence.h5"),
    )


def test_cache_round_trip_preserves_arrays_and_orders(tmp_path):
    cache_path = tmp_path / "prepared.h5"
    prepared = _prepared(tmp_path)
    signature = {"schema": 1, "input_identity": "abc"}

    save_prepared_cache(cache_path, prepared, signature)
    loaded = load_prepared_cache(cache_path, expected_signature=signature)

    assert loaded.subcluster_names == prepared.subcluster_names
    assert loaded.expression_genes == prepared.expression_genes
    assert loaded.train_genes == prepared.train_genes
    assert loaded.validation_genes == prepared.validation_genes
    assert loaded.test_genes == prepared.test_genes
    assert loaded.cell_type_to_id == prepared.cell_type_to_id
    assert loaded.lr_feature_names == prepared.lr_feature_names
    assert loaded.gene_weights == prepared.gene_weights
    assert loaded.sequence_hdf5_path == prepared.sequence_hdf5_path
    np.testing.assert_array_equal(loaded.major_ids, prepared.major_ids)
    np.testing.assert_array_equal(loaded.mean_environment, prepared.mean_environment)
    np.testing.assert_array_equal(loaded.mean_receiver_ccc, prepared.mean_receiver_ccc)
    np.testing.assert_array_equal(loaded.mean_expression, prepared.mean_expression)


def test_cache_rejects_mismatched_signature(tmp_path):
    cache_path = tmp_path / "prepared.h5"
    save_prepared_cache(cache_path, _prepared(tmp_path), {"schema": 1})
    with pytest.raises(CacheError, match="signature"):
        load_prepared_cache(cache_path, expected_signature={"schema": 2})

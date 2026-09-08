from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genospatial.dataset import GenoSpatialDataset, build_dataloaders
from genospatial.preprocessing import PreparedTrainingData


def _prepared(tmp_path: Path) -> PreparedTrainingData:
    sequence_path = tmp_path / "sequence.h5"
    with h5py.File(sequence_path, "w") as handle:
        handle.create_dataset("g1", data=np.arange(12).reshape(3, 4))
        handle.create_dataset("g2", data=np.arange(12, 24).reshape(3, 4))
    return PreparedTrainingData(
        subcluster_names=["A_n1", "B_n2"],
        major_ids=np.array([0, 1]),
        mean_environment=np.array([[1.0, 2.0], [3.0, 4.0]]),
        mean_receiver_ccc=np.array([[5.0], [6.0]]),
        mean_expression=np.array([[0.1, 0.2], [0.3, 0.4]]),
        expression_genes=["g1", "g2"],
        train_genes=["g1"],
        validation_genes=["g2"],
        test_genes=["g2"],
        gene_weights={"g1": 1.5, "g2": 2.0},
        cell_type_to_id={"A": 0, "B": 1},
        lr_feature_names=["lr_a"],
        sequence_hdf5_path=str(sequence_path),
    )


def test_dataset_enumerates_subcluster_major_and_returns_notebook_tuple(tmp_path):
    prepared = _prepared(tmp_path)
    dataset = GenoSpatialDataset(prepared, ["g1", "g2"], prepared.gene_weights)

    assert dataset.samples == [
        ("A_n1", "g1"),
        ("A_n1", "g2"),
        ("B_n2", "g1"),
        ("B_n2", "g2"),
    ]
    sequence, cell_id, environment, ccc, target, weight, gene = dataset[1]
    assert sequence.dtype == torch.float32
    assert cell_id.dtype == torch.long
    assert environment.dtype == torch.float32
    assert ccc.dtype == torch.float32
    assert target.dtype == torch.float32
    assert weight.dtype == torch.float32
    assert gene == "g2"
    assert target.item() == pytest.approx(0.2)
    assert weight.item() == pytest.approx(2.0)


def test_dataset_defaults_weight_to_one(tmp_path):
    prepared = _prepared(tmp_path)
    dataset = GenoSpatialDataset(prepared, ["g2"], gene_weights=None)
    assert dataset[0][5].item() == 1.0


def test_dataloader_settings_match_notebook(tmp_path):
    prepared = _prepared(tmp_path)
    config = {
        "training": {
            "batch_size": 256,
            "validation_batch_size": 256,
            "test_batch_size": 128,
            "train_workers": 4,
            "validation_workers": 0,
            "test_workers": 0,
        }
    }
    train, validation, test = build_dataloaders(prepared, config)
    assert train.batch_size == 256 and train.num_workers == 4
    assert validation.batch_size == 256 and validation.num_workers == 0
    assert test.batch_size == 128 and test.num_workers == 0
    assert train.sampler.__class__.__name__ == "RandomSampler"
    assert validation.sampler.__class__.__name__ == "SequentialSampler"

"""Dataset and DataLoader definitions matching the authoritative notebook."""

from __future__ import annotations

from typing import Mapping, Sequence

import h5py
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import PreparedTrainingData


class GenoSpatialDataset(Dataset):
    """Cartesian product of retained subclusters and a chromosome gene split."""

    def __init__(
        self,
        prepared: PreparedTrainingData,
        genes: Sequence[str],
        gene_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.prepared = prepared
        self.genes = list(genes)
        self.subclusters = list(prepared.subcluster_names)
        self.samples = [
            (subcluster, gene)
            for subcluster in self.subclusters
            for gene in self.genes
        ]
        self.gene_weights = gene_weights
        self.h5_path = prepared.sequence_hdf5_path
        self._subcluster_to_row = {
            name: index for index, name in enumerate(prepared.subcluster_names)
        }
        self._gene_to_column = {
            gene: index for index, gene in enumerate(prepared.expression_genes)
        }
        missing = [gene for gene in self.genes if gene not in self._gene_to_column]
        if missing:
            raise KeyError(f"Prepared expression matrix is missing gene: {missing[0]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        subcluster, gene = self.samples[index]
        row = self._subcluster_to_row[subcluster]
        column = self._gene_to_column[gene]

        cell_id = torch.tensor(self.prepared.major_ids[row], dtype=torch.long)
        environment = torch.tensor(
            self.prepared.mean_environment[row], dtype=torch.float32
        )
        receiver_ccc = torch.tensor(
            self.prepared.mean_receiver_ccc[row], dtype=torch.float32
        )
        target = torch.tensor(
            self.prepared.mean_expression[row, column], dtype=torch.float32
        )
        with h5py.File(self.h5_path, "r") as handle:
            sequence = torch.tensor(handle[gene][:], dtype=torch.float32)
        weight_value = (
            self.gene_weights.get(gene, 1.0)
            if self.gene_weights is not None
            else 1.0
        )
        weight = torch.tensor(weight_value, dtype=torch.float32)
        return sequence, cell_id, environment, receiver_ccc, target, weight, gene


def build_dataloaders(
    prepared: PreparedTrainingData, config: Mapping
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct split loaders with the frozen batch/shuffle/worker behavior."""

    training = config["training"]
    train_dataset = GenoSpatialDataset(
        prepared, prepared.train_genes, prepared.gene_weights
    )
    validation_dataset = GenoSpatialDataset(
        prepared, prepared.validation_genes, prepared.gene_weights
    )
    test_dataset = GenoSpatialDataset(prepared, prepared.test_genes, None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=training["train_workers"],
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training["validation_batch_size"],
        shuffle=False,
        num_workers=training["validation_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training["test_batch_size"],
        shuffle=False,
        num_workers=training["test_workers"],
    )
    return train_loader, validation_loader, test_loader

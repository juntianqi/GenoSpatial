from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genospatial.training import EarlyStopping, train, validate, weighted_huber_loss


def test_weighted_huber_matches_explicit_formula_and_backward():
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    features = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    targets = torch.tensor([0.2, 0.8])
    weights = torch.tensor([1.0, 3.0])

    before = model.weight.detach().clone()
    predictions = model(features).squeeze(-1)
    per_sample = torch.nn.functional.huber_loss(
        predictions, targets, reduction="none", delta=0.1
    )
    expected = (per_sample * weights).mean()
    observed = weighted_huber_loss(predictions, targets, weights, delta=0.1)
    torch.testing.assert_close(observed, expected)

    optimizer.zero_grad()
    observed.backward()
    assert model.weight.grad is not None
    optimizer.step()
    assert not torch.equal(model.weight.detach(), before)


def test_early_stopping_saves_first_and_qualifying_lower_loss(tmp_path):
    model = nn.Linear(2, 1)
    checkpoint = tmp_path / "model.pth"
    stopper = EarlyStopping(
        patience=2, minimum_delta=0.1, checkpoint_path=checkpoint
    )

    stopper(1.0, model)
    assert checkpoint.is_file()
    assert stopper.best_loss == 1.0

    with torch.no_grad():
        model.weight.add_(1.0)
    stopper(0.95, model)
    assert stopper.counter == 1
    assert not stopper.early_stop

    stopper(0.90, model)
    assert stopper.counter == 0
    assert stopper.best_loss == pytest.approx(0.90)

    stopper(0.85, model)
    stopper(0.84, model)
    assert stopper.early_stop


class ValidationDataset(Dataset):
    def __init__(self):
        self.targets = torch.tensor([0.0, 1.0, 2.0])
        self.weights = torch.tensor([1.0, 2.0, 3.0])

    def __len__(self):
        return 3

    def __getitem__(self, index):
        value = torch.tensor([float(index)])
        return (
            value.reshape(1, 1),
            torch.tensor(0),
            value,
            value,
            self.targets[index],
            self.weights[index],
            f"g{index}",
        )


class EchoModel(nn.Module):
    def forward(self, sequence, cell_ids, environment, ccc):
        return environment.squeeze(-1)


def test_validate_uses_sample_weighted_batch_loss_and_global_pearson():
    loader = DataLoader(ValidationDataset(), batch_size=2, shuffle=False)
    result = validate(EchoModel(), loader, torch.device("cpu"), delta=0.1)
    # Predictions equal targets, so every Huber term is zero and Pearson is one.
    assert result.loss == 0.0
    assert result.pearson_r == pytest.approx(1.0)
    np.testing.assert_array_equal(result.predictions, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(result.targets, [0.0, 1.0, 2.0])


class ConstantModel(nn.Module):
    def forward(self, sequence, cell_ids, environment, ccc):
        return torch.zeros(sequence.size(0))


def test_validate_returns_zero_pearson_for_constant_predictions():
    loader = DataLoader(ValidationDataset(), batch_size=2, shuffle=False)
    result = validate(ConstantModel(), loader, torch.device("cpu"), delta=0.1)
    assert result.pearson_r == 0.0


class DeviceOrderTrackingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.events = []
        self.moved_to_device = False

    def to(self, *args, **kwargs):
        self.events.append("to")
        result = super().to(*args, **kwargs)
        self.moved_to_device = True
        return result

    def parameters(self, recurse=True):
        self.events.append("parameters")
        if not self.moved_to_device:
            raise AssertionError("optimizer consumed parameters before model.to(device)")
        return super().parameters(recurse=recurse)


def test_train_moves_model_to_device_before_optimizer_construction(tmp_path):
    model = DeviceOrderTrackingModel()
    config = {
        "training": {"epochs": 1, "validation_every_steps": 250},
        "optimizer": {"learning_rate": 5e-5, "weight_decay": 1e-3},
        "loss": {"delta": 0.1},
        "early_stopping": {"patience": 10, "minimum_delta": 1e-4},
        "output": {
            "directory": str(tmp_path),
            "checkpoint_name": "model.pth",
        },
    }

    train(model, [], [], config, torch.device("cpu"))

    assert model.events[:2] == ["to", "parameters"]

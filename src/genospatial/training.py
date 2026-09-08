"""Frozen v12_r optimization, validation and checkpoint behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .model import GenoSpatialModel


@dataclass
class ValidationResult:
    loss: float
    pearson_r: float
    predictions: np.ndarray
    targets: np.ndarray


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    validation_correlations: list[float] = field(default_factory=list)
    global_steps: list[int] = field(default_factory=list)


def weighted_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    """Huber per item, multiplied by gene weight, followed by mean."""

    criterion = nn.HuberLoss(reduction="none", delta=delta)
    unweighted_loss = criterion(predictions, targets)
    return (unweighted_loss * weights).mean()


def _pearson_correlation(targets: np.ndarray, predictions: np.ndarray) -> float:
    if np.std(targets) == 0 or np.std(predictions) == 0:
        return 0.0
    try:
        from scipy.stats import pearsonr

        result = pearsonr(targets, predictions)
        return float(result.statistic if hasattr(result, "statistic") else result[0])
    except ImportError:
        return float(np.corrcoef(targets, predictions)[0, 1])


def validate(
    model: nn.Module,
    validation_loader,
    device: torch.device,
    *,
    delta: float = 0.1,
) -> ValidationResult:
    """Run the notebook's global validation-loss and Pearson calculation."""

    model.eval()
    validation_loss = 0.0
    all_predictions: list[float] = []
    all_targets: list[float] = []
    with torch.no_grad():
        for (
            batch_sequence,
            batch_cell,
            batch_environment,
            batch_ccc,
            batch_targets,
            batch_weights,
            _,
        ) in validation_loader:
            batch_sequence = batch_sequence.to(device)
            batch_cell = batch_cell.to(device)
            batch_environment = batch_environment.to(device)
            batch_ccc = batch_ccc.to(device)
            batch_targets = batch_targets.to(device)
            batch_weights = batch_weights.to(device)

            predictions = model(
                batch_sequence, batch_cell, batch_environment, batch_ccc
            )
            loss = weighted_huber_loss(
                predictions, batch_targets, batch_weights, delta=delta
            )
            validation_loss += loss.item() * batch_targets.size(0)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())

    validation_loss /= len(validation_loader.dataset)
    prediction_array = np.asarray(all_predictions)
    target_array = np.asarray(all_targets)
    correlation = _pearson_correlation(target_array, prediction_array)
    return ValidationResult(
        loss=float(validation_loss),
        pearson_r=correlation,
        predictions=prediction_array,
        targets=target_array,
    )


class EarlyStopping:
    """Validation-loss stopping and state-dict checkpointing from cell 40."""

    def __init__(
        self,
        patience: int = 5,
        minimum_delta: float = 1e-4,
        checkpoint_path: str | Path = "best_Advtrimodal_model_v12_r.pth",
    ) -> None:
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.counter = 0
        self.best_loss: float | None = None
        self.early_stop = False

    def __call__(self, validation_loss: float, model: nn.Module) -> None:
        if self.best_loss is None:
            self.best_loss = validation_loss
            self.save_checkpoint(model)
        elif validation_loss > self.best_loss - self.minimum_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = validation_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model: nn.Module) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        source_model = model.module if hasattr(model, "module") else model
        torch.save(source_model.state_dict(), self.checkpoint_path)


def resolve_device(device_setting: str) -> torch.device:
    """Resolve a portable device while retaining cuda:1 when available."""

    if device_setting != "auto":
        return torch.device(device_setting)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if torch.cuda.device_count() > 1:
        return torch.device("cuda:1")
    return torch.device("cuda:0")


def build_model(prepared, config: Mapping[str, Any]) -> GenoSpatialModel:
    """Instantiate the frozen architecture with data-derived cardinalities."""

    model_config = config["model"]
    return GenoSpatialModel(
        seq_len=model_config["seq_len"],
        seq_channels=model_config["seq_channels"],
        num_major_types=len(prepared.cell_type_to_id),
        cell_embed_dim=model_config["cell_embed_dim"],
        env_dim=model_config["environment_dim"],
        ccc_dim=prepared.mean_receiver_ccc.shape[1],
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        num_factors=model_config["num_factors"],
        branch_dropout_probability=model_config["branch_dropout_probability"],
        predictor_dropout_probability=model_config["predictor_dropout_probability"],
    )


def train(
    model: nn.Module,
    train_loader,
    validation_loader,
    config: Mapping[str, Any],
    device: torch.device,
) -> TrainingHistory:
    """Train with global-step validation and frozen early-stopping semantics."""

    training_config = config["training"]
    optimizer_config = config["optimizer"]
    loss_delta = config["loss"]["delta"]
    output_directory = Path(config["output"]["directory"])
    checkpoint_path = output_directory / config["output"]["checkpoint_name"]
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
    )
    early_stopper = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        minimum_delta=config["early_stopping"]["minimum_delta"],
        checkpoint_path=checkpoint_path,
    )
    history = TrainingHistory()
    global_step = 0

    for epoch in range(1, training_config["epochs"] + 1):
        model.train()
        cumulative_train_loss = 0.0
        samples_seen = 0
        for (
            batch_sequence,
            batch_cell,
            batch_environment,
            batch_ccc,
            batch_targets,
            batch_weights,
            _,
        ) in train_loader:
            batch_sequence = batch_sequence.to(device)
            batch_cell = batch_cell.to(device)
            batch_environment = batch_environment.to(device)
            batch_ccc = batch_ccc.to(device)
            batch_targets = batch_targets.to(device)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad()
            predictions = model(
                batch_sequence, batch_cell, batch_environment, batch_ccc
            )
            loss = weighted_huber_loss(
                predictions, batch_targets, batch_weights, delta=loss_delta
            )
            loss.backward()
            optimizer.step()

            cumulative_train_loss += loss.item() * batch_targets.size(0)
            samples_seen += batch_targets.size(0)
            global_step += 1

            if global_step % training_config["validation_every_steps"] == 0:
                average_train_loss = cumulative_train_loss / samples_seen
                result = validate(
                    model, validation_loader, device, delta=loss_delta
                )
                history.train_losses.append(float(average_train_loss))
                history.validation_losses.append(result.loss)
                history.validation_correlations.append(result.pearson_r)
                history.global_steps.append(global_step)
                print(
                    f"Epoch {epoch:03d} step {global_step}: "
                    f"weighted train Huber={average_train_loss:.4f}, "
                    f"weighted validation Huber={result.loss:.4f}, "
                    f"validation Pearson r={result.pearson_r:.4f}"
                )
                early_stopper(result.loss, model)
                if early_stopper.early_stop:
                    print(
                        "Early stopping after "
                        f"{early_stopper.patience} non-improving validation events"
                    )
                    break
                model.train()
        if early_stopper.early_stop:
            break
    return history

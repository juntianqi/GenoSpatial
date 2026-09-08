from pathlib import Path
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genospatial.model import GenoSpatialModel


class NotebookReferenceModel(nn.Module):
    """Compact literal reference for authoritative notebook cell 41."""

    def __init__(
        self,
        seq_channels=4,
        num_major_types=3,
        cell_embed_dim=8,
        env_dim=6,
        ccc_dim=5,
        hidden_dim=8,
        num_heads=2,
        num_factors=2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_factors = num_factors
        self.seq_conv = nn.Sequential(
            nn.Conv1d(seq_channels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.cell_embedding = nn.Embedding(num_major_types, cell_embed_dim)
        self.film_generator = nn.Sequential(
            nn.Linear(cell_embed_dim, hidden_dim * 2), nn.GELU()
        )
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, hidden_dim * num_factors),
            nn.LayerNorm(hidden_dim * num_factors),
        )
        self.ccc_proj = nn.Sequential(
            nn.Linear(ccc_dim, hidden_dim * num_factors),
            nn.LayerNorm(hidden_dim * num_factors),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ccc_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.saved_env_attn_weights = None
        self.saved_ccc_attn_weights = None

    def forward(self, seq_feat, major_cell_idx, env_feat, ccc_feat):
        seq_feat = seq_feat.permute(0, 2, 1)
        z_seq = self.seq_conv(seq_feat)
        z_cell = self.cell_embedding(major_cell_idx)
        film_params = self.film_generator(z_cell)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        z_seq_film = z_seq * (1.0 + gamma) + beta
        z_seq_film = z_seq_film.permute(0, 2, 1)
        z_baseline = z_seq_film.mean(dim=1)
        batch_size = z_seq_film.size(0)
        z_env_flat = self.env_proj(env_feat)
        z_ccc_flat = self.ccc_proj(ccc_feat)
        q_env = z_env_flat.view(batch_size, self.num_factors, self.hidden_dim)
        q_ccc = z_ccc_flat.view(batch_size, self.num_factors, self.hidden_dim)
        env_output, env_weights = self.cross_attn(q_env, z_seq_film, z_seq_film)
        ccc_output, ccc_weights = self.ccc_attn(q_ccc, z_seq_film, z_seq_film)
        if not self.training:
            self.saved_env_attn_weights = env_weights.detach()
            self.saved_ccc_attn_weights = ccc_weights.detach()
        env_output = env_output.sum(dim=1)
        ccc_output = ccc_output.sum(dim=1)
        z_final = self.fusion_norm(z_baseline + env_output + ccc_output)
        return self.predictor(z_final).squeeze(-1)


def _small_model():
    return GenoSpatialModel(
        seq_len=7,
        seq_channels=4,
        num_major_types=3,
        cell_embed_dim=8,
        env_dim=6,
        ccc_dim=5,
        hidden_dim=8,
        num_heads=2,
        num_factors=2,
        branch_dropout_probability=0.25,
        predictor_dropout_probability=0.2,
    )


def test_frozen_default_layer_shapes():
    model = GenoSpatialModel(num_major_types=20, ccc_dim=1008)
    assert model.seq_conv[0].weight.shape == (512, 1920, 5)
    assert model.seq_conv[3].weight.shape == (128, 512, 3)
    assert model.env_proj[0].out_features == 128 * 8
    assert model.ccc_proj[0].in_features == 1008
    assert model.ccc_proj[0].out_features == 128 * 8
    assert model.predictor[-1].out_features == 1


def test_eval_forward_and_state_dict_are_notebook_equivalent():
    torch.manual_seed(7)
    reference = NotebookReferenceModel()
    public = _small_model()
    assert list(public.state_dict()) == list(reference.state_dict())
    public.load_state_dict(reference.state_dict())
    reference.eval()
    public.eval()

    sequence = torch.randn(2, 7, 4)
    cell_ids = torch.tensor([0, 2])
    environment = torch.randn(2, 6)
    ccc = torch.randn(2, 5)
    expected = reference(sequence, cell_ids, environment, ccc)
    observed = public(sequence, cell_ids, environment, ccc)

    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(
        public.saved_env_attn_weights, reference.saved_env_attn_weights
    )
    torch.testing.assert_close(
        public.saved_ccc_attn_weights, reference.saved_ccc_attn_weights
    )


def test_branch_dropout_masks_are_independent_per_sample(monkeypatch):
    model = _small_model()
    environment = torch.ones(3, 8)
    ccc = torch.full((3, 8), 2.0)
    masks = iter(
        [
            torch.tensor([[0.1], [0.9], [0.9]]),
            torch.tensor([[0.9], [0.1], [0.9]]),
        ]
    )
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(masks))

    environment_result, ccc_result = model._drop_context_branches(environment, ccc)

    assert torch.equal(environment_result[0], torch.zeros(8))
    assert torch.equal(environment_result[1], torch.ones(8))
    assert torch.equal(ccc_result[0], torch.full((8,), 2.0))
    assert torch.equal(ccc_result[1], torch.zeros(8))
    assert torch.equal(environment_result[2], torch.ones(8))
    assert torch.equal(ccc_result[2], torch.full((8,), 2.0))


def test_full_model_single_minibatch_backward_and_optimizer_step():
    torch.manual_seed(11)
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

    prediction = model(
        torch.randn(2, 7, 4),
        torch.tensor([0, 2]),
        torch.randn(2, 6),
        torch.randn(2, 5),
    )
    target = torch.randn(2)
    weight = torch.tensor([1.0, 2.0])
    loss = (
        torch.nn.functional.huber_loss(
            prediction, target, delta=0.1, reduction="none"
        )
        * weight
    ).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

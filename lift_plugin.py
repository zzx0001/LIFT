import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

PLUGIN_DIR = Path(__file__).resolve().parent
CODE_DIR = PLUGIN_DIR.parent
for path in (PLUGIN_DIR, CODE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lift_core import CausalContinuousGaborTokenizer, SemanticsEmbedding


class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x, mask=None):
        x = x.clamp(min=self.eps)
        if mask is None:
            return x.pow(self.p).mean(dim=1).pow(1.0 / self.p)

        valid = mask.float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=self.eps)
        pooled = (x.pow(self.p) * valid).sum(dim=1) / denom
        return pooled.pow(1.0 / self.p)


class LIFTFrequencyPlugin(nn.Module):
    """Frequency-only LIFT module that can be attached to external backbones."""

    def __init__(
        self,
        num_types,
        emb_dim=32,
        num_atoms=24,
        grid_size=48,
        max_time=12,
        dc=True,
        use_phase=False,
        trend_period_min=4.0,
        trend_period_max=12.0,
        event_period_min=1.0,
        event_period_max=6.0,
        trend_beta_init=5.0,
        event_beta_init=2.0,
        trust_temperature_init=3.0,
        pooling="mean",
        attn_hidden_dim=128,
        dropout=0.1,
        delta_t_mode="semantic",
        use_ftm=True,
        use_density_comp=True,
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.grid_size = int(grid_size)
        self.max_time = float(max_time)
        self.pooling = pooling.lower()
        self.dc = bool(dc)
        self.use_phase = bool(use_phase)
        self.delta_t_mode = delta_t_mode.lower()
        if self.delta_t_mode not in {"semantic", "time_step", "feature"}:
            raise ValueError(f"Unknown delta_t_mode: {delta_t_mode}")

        self.embedding = SemanticsEmbedding(
            num_types=self.num_types,
            input_dim=1,
            hidden_dim=emb_dim,
        )
        self.tokenizer = CausalContinuousGaborTokenizer(
            input_dim=emb_dim,
            num_atoms=num_atoms,
            grid_size=grid_size,
            max_time=max_time,
            dc=dc,
            use_phase=use_phase,
            trend_period_min=trend_period_min,
            trend_period_max=trend_period_max,
            event_period_min=event_period_min,
            event_period_max=event_period_max,
            trend_beta_init=trend_beta_init,
            event_beta_init=event_beta_init,
            trust_temperature_init=trust_temperature_init,
            use_ftm=use_ftm,
            use_density_comp=use_density_comp,
        )

        multiplier = 2 if use_phase else 1
        self.token_dim = num_atoms * emb_dim * multiplier + (emb_dim if dc else 0)
        self.summary_multiplier = 1
        if self.pooling == "gem":
            self.pool = GeMPooling()
        elif self.pooling in {"mean", "max"}:
            self.pool = None
        elif self.pooling == "attn":
            self.pool = None
            self.attn_pool = nn.Sequential(
                nn.Linear(self.token_dim, attn_hidden_dim),
                nn.Tanh(),
                nn.Linear(attn_hidden_dim, 1),
            )
        elif self.pooling in {"mean_max", "mean_attn", "mean_max_attn"}:
            self.pool = None
            if "attn" in self.pooling:
                self.attn_pool = nn.Sequential(
                    nn.Linear(self.token_dim, attn_hidden_dim),
                    nn.Tanh(),
                    nn.Linear(attn_hidden_dim, 1),
                )
            self.summary_multiplier = len(self.pooling.split("_"))
        else:
            raise ValueError(f"Unknown plugin pooling: {pooling}")
        self.dropout = nn.Dropout(dropout)

    def _time_step_delta_t(self, x, x_mask, time_idx):
        del x_mask
        n_steps = x.shape[0]
        if n_steps <= 1:
            step_delta = x.new_ones((1,))
        else:
            step_delta = x.new_ones((n_steps,))
            step_delta[1:] = 1.0
        return step_delta[time_idx]

    def _feature_delta_t(self, x, x_mask, time_idx, feat_idx):
        del x
        deltas = torch.ones_like(time_idx, dtype=x_mask.dtype)
        for feat in torch.unique(feat_idx):
            positions = torch.nonzero(feat_idx == feat, as_tuple=False).flatten()
            feat_times = time_idx[positions].to(dtype=x_mask.dtype)
            if positions.numel() <= 1:
                deltas[positions] = 1.0
                continue

            feat_deltas = torch.empty_like(feat_times)
            feat_deltas[1:] = feat_times[1:] - feat_times[:-1]
            valid_deltas = feat_deltas[1:].clamp(min=1.0)
            feat_deltas[0] = valid_deltas.mean() if valid_deltas.numel() > 0 else 1.0
            deltas[positions] = feat_deltas.clamp(min=1.0)
        return deltas

    def _observation_delta_t(self, x, x_mask, time_idx, feat_idx):
        if self.delta_t_mode == "semantic":
            return None
        if self.delta_t_mode == "time_step":
            return self._time_step_delta_t(x, x_mask, time_idx)
        return self._feature_delta_t(x, x_mask, time_idx, feat_idx)

    def _dense_to_padded_observations(self, x, x_mask=None):
        """Convert dense x/mask tensors to padded value-time-type observations."""
        if x_mask is None:
            x_mask = torch.ones_like(x)

        batch_size, n_steps, n_features = x.shape
        device = x.device
        time_grid = torch.arange(n_steps, dtype=x.dtype, device=device)

        values_list = []
        times_list = []
        types_list = []
        delta_t_list = []
        lengths = []
        for i in range(batch_size):
            valid = x_mask[i].bool()
            time_idx, feat_idx = valid.nonzero(as_tuple=True)
            if time_idx.numel() == 0:
                time_idx = torch.zeros(1, dtype=torch.long, device=device)
                feat_idx = torch.zeros(1, dtype=torch.long, device=device)
            values_list.append(x[i, time_idx, feat_idx].unsqueeze(-1))
            times_list.append(time_grid[time_idx])
            types_list.append(feat_idx.long().clamp(max=self.num_types - 1))
            delta_t = self._observation_delta_t(x[i], x_mask[i], time_idx, feat_idx)
            if delta_t is not None:
                delta_t_list.append(delta_t)
            lengths.append(time_idx.numel())

        max_len = max(lengths)
        values = x.new_zeros((batch_size, max_len, 1))
        times = x.new_zeros((batch_size, max_len))
        types = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        mask = x.new_zeros((batch_size, max_len))
        delta_t_out = None
        if self.delta_t_mode != "semantic":
            delta_t_out = x.new_zeros((batch_size, max_len))
        for i, length in enumerate(lengths):
            values[i, :length] = values_list[i]
            times[i, :length] = times_list[i]
            types[i, :length] = types_list[i]
            mask[i, :length] = 1.0
            if delta_t_out is not None:
                delta_t_out[i, :length] = delta_t_list[i]

        return values, times, types, mask, delta_t_out

    def _grid_mask(self, times, obs_mask):
        actual_max_time = (times * obs_mask).max(dim=1).values
        grid_points = self.tokenizer.grid_points
        return grid_points.unsqueeze(0) <= actual_max_time.unsqueeze(1)

    @staticmethod
    def _masked_mean(tokens, grid_mask):
        valid = grid_mask.float().unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _masked_max(tokens, grid_mask):
        return tokens.masked_fill(~grid_mask.unsqueeze(-1), -1e9).max(dim=1).values

    def _masked_attn(self, tokens, grid_mask):
        scores = self.attn_pool(tokens).squeeze(-1)
        scores = scores.masked_fill(~grid_mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (tokens * weights).sum(dim=1)

    def _pool_tokens(self, tokens, grid_mask):
        if self.pooling == "mean":
            return self._masked_mean(tokens, grid_mask)
        if self.pooling == "max":
            return self._masked_max(tokens, grid_mask)
        if self.pooling == "gem":
            return self.pool(tokens, grid_mask)
        if self.pooling == "attn":
            return self._masked_attn(tokens, grid_mask)
        if self.pooling == "mean_max":
            return torch.cat([
                self._masked_mean(tokens, grid_mask),
                self._masked_max(tokens, grid_mask),
            ], dim=-1)
        if self.pooling == "mean_attn":
            return torch.cat([
                self._masked_mean(tokens, grid_mask),
                self._masked_attn(tokens, grid_mask),
            ], dim=-1)
        if self.pooling == "mean_max_attn":
            return torch.cat([
                self._masked_mean(tokens, grid_mask),
                self._masked_max(tokens, grid_mask),
                self._masked_attn(tokens, grid_mask),
            ], dim=-1)
        raise RuntimeError(f"Unsupported plugin pooling: {self.pooling}")

    def forward(self, x, x_mask=None):
        values, times, types, obs_mask, external_delta_t = self._dense_to_padded_observations(x, x_mask)
        h, delta_t = self.embedding(values, times, types, obs_mask, delta_t=external_delta_t)

        if delta_t.shape[1] > 0 and delta_t[:, 0].sum() < 1e-2:
            valid_len = obs_mask.sum(dim=1, keepdim=True).clamp(min=1)
            total_duration = (times * obs_mask).max(dim=1, keepdim=True).values
            delta_t[:, 0] = (total_duration / valid_len).squeeze(-1)

        tokens = self.tokenizer(h, times, delta_t, obs_mask)
        grid_mask = self._grid_mask(times, obs_mask)
        summary = self._pool_tokens(tokens, grid_mask)

        return self.dropout(summary)


class LIFTPluginHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_classes=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)

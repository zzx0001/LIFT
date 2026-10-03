import torch
import torch.nn as nn

from lift_plugin import LIFTFrequencyPlugin


class UncertaintyAwareSafeFusion(nn.Module):
    """Do-no-harm fusion: at t=0, final_logits == base_logits by construction.

    At init: `plugin_to_logit` weight and bias are zero → delta = 0 →
    final_logits = base_logits regardless of plugin_summary or gate.

    `gate = sigmoid(k0 + k1 * confidence)`. `confidence` = detached
    absolute logit margin (top1 − top2 for K-class softmax; |logit| for
    1-logit binary). With defaults `k0=-3, k1=-0.5`, the gate is small
    when confidence is high (~0.047 at zero confidence; smaller when
    confident). After training, k0/k1 can adapt to widen the gate where
    the plug-in is helpful.

    Args:
        plugin_dim: dimensionality of `plugin_summary`. For the default
            LIFT plug-in config (`num_atoms=24, emb_dim=32, dc=True,
            use_phase=False`) this is 800.
        num_classes: output logits dim. Typically 2 for binary heads.
        k0_init, k1_init: gate parameter inits.
    """

    def __init__(self, plugin_dim: int, num_classes: int = 2,
                 k0_init: float = -3.0, k1_init: float = -0.5,
                 no_zero_init: bool = False, no_gate: bool = False):
        super().__init__()
        self.plugin_to_logit = nn.Linear(plugin_dim, num_classes)
        if no_zero_init:
            # Ablation: xavier_uniform_ init instead of zeros. Breaks
            # do-no-harm at t=0; the residual is nonzero from step 0.
            nn.init.xavier_uniform_(self.plugin_to_logit.weight)
            nn.init.zeros_(self.plugin_to_logit.bias)
        else:
            nn.init.zeros_(self.plugin_to_logit.weight)
            nn.init.zeros_(self.plugin_to_logit.bias)
        self.k0 = nn.Parameter(torch.tensor(float(k0_init)))
        self.k1 = nn.Parameter(torch.tensor(float(k1_init)))
        self.no_zero_init = bool(no_zero_init)
        self.no_gate = bool(no_gate)

    @staticmethod
    def _confidence(base_logits: torch.Tensor) -> torch.Tensor:
        """Non-negative confidence proxy of shape (B,).

        - 1-D logits (B,): |logit|.
        - 1-logit binary (B, 1): |logit|.
        - K-class softmax (B, K): top-1 margin |top1 − top2|.
        """
        x = base_logits.detach()
        if x.dim() == 1:
            return x.abs()
        if x.size(-1) == 1:
            return x.squeeze(-1).abs()
        top2, _ = x.topk(2, dim=-1)
        return (top2[..., 0] - top2[..., 1]).abs()

    def forward(self, base_logits: torch.Tensor,
                plugin_summary: torch.Tensor) -> torch.Tensor:
        delta = self.plugin_to_logit(plugin_summary)
        if delta.shape != base_logits.shape:
            delta = delta.view_as(base_logits)
        if self.no_gate:
            # Ablation: disable uncertainty-aware gate (gate ≡ 1).
            gate = torch.ones(base_logits.shape[0],
                              device=delta.device, dtype=delta.dtype)
        else:
            conf = self._confidence(base_logits)              # (B,)
            gate = torch.sigmoid(self.k0 + self.k1 * conf)    # (B,)
        while gate.dim() < delta.dim():
            gate = gate.unsqueeze(-1)
        return base_logits + gate * delta

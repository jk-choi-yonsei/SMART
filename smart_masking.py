"""Self-tuning action masking with adaptive reliability threshold (SMART).

Reference implementation of the masking components described in the paper:

- VRNet: predicts, for every discrete action, its validity and the reliability
  of that prediction (Section 4.1).
- Adaptive masking (AMI): hard, no, or soft masking of the policy logits
  depending on the predicted validity and reliability (Section 4.2).
- Performance-driven curriculum: the masking is tightened only after the
  agent reliably completes the episode under the current setting (Section 4.2).
- Dual-head actor: a shared encoder with one policy head per robot (Section 4.3).

The module only depends on PyTorch and can be combined with any discrete-action
RL algorithm (e.g. SAC or PPO) whose environment provides
(i) a mask of physically feasible actions and (ii) validity labels for
training the VRNet.
"""

from collections import deque
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: Sequence[int]) -> Tuple[nn.Sequential, int]:
    layers, dim = [], in_dim
    for h in hidden:
        layers += [nn.Linear(dim, h), nn.ReLU()]
        dim = h
    return nn.Sequential(*layers), dim


class VRNet(nn.Module):
    """Validity-reliability network.

    A shared feature extractor followed by two heads. For a batch of states it
    returns the validity probability and the reliability of that prediction for
    every action, both in [0, 1].
    """

    def __init__(self, state_dim: int, num_actions: int, hidden: Sequence[int] = (256, 128)):
        super().__init__()
        self.body, feat_dim = _mlp(state_dim, hidden)
        self.validity_head = nn.Linear(feat_dim, num_actions)
        self.reliability_head = nn.Linear(feat_dim, num_actions)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(state)
        validity = torch.sigmoid(self.validity_head(h))
        reliability = torch.sigmoid(self.reliability_head(h))
        return validity, reliability


def vrnet_loss(
    validity: torch.Tensor,
    reliability: torch.Tensor,
    validity_label: torch.Tensor,
    lam: float = 0.1,
    label_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-task loss L = L_validity + lam * L_reliability (Eqs. in Section 4.1).

    validity, reliability: VRNet outputs, shape (batch, num_actions).
    validity_label: 1 if the action is logically valid, 0 otherwise.
    label_mask: optional, 1 where a label is available (e.g. only physically
        feasible actions); other entries are ignored.

    The reliability target is 1 where the validity prediction is correct
    (threshold 0.5) and 0 otherwise.
    """
    validity_label = validity_label.float()
    if label_mask is None:
        label_mask = torch.ones_like(validity_label)
    label_mask = label_mask.float()

    bce = F.binary_cross_entropy(validity.clamp(1e-6, 1 - 1e-6), validity_label, reduction="none")
    loss_validity = (bce * label_mask).sum(dim=-1).mean()

    with torch.no_grad():
        correct = ((validity > 0.5).float() == validity_label).float()
    mse = (reliability - correct) ** 2
    loss_reliability = (mse * label_mask).sum(dim=-1).mean()

    return loss_validity + lam * loss_reliability


def adaptive_mask(
    logits: torch.Tensor,
    validity: torch.Tensor,
    reliability: torch.Tensor,
    tau_r: float,
    alpha_soft: float,
    feasible: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Apply the adaptive masking index (AMI) to policy logits.

    Case 1 (predicted invalid and reliable, v < 0.5 and r >= tau_r): hard mask.
    Case 2 (predicted valid, v >= 0.5): no change.
    Case 3 (predicted invalid but unreliable, v < 0.5 and r < tau_r):
        logit -= alpha_soft * (0.5 - v) * (1 - r).

    feasible: optional boolean mask of physically feasible actions (L1 mask);
        infeasible actions are always removed.
    dropout: probability that the learned mask is ignored for each action
        (the mask dropout rate epsilon used during training; 0 at evaluation).
    """
    predicted_invalid = validity < 0.5
    hard = predicted_invalid & (reliability >= tau_r)
    soft = predicted_invalid & (reliability < tau_r)
    penalty = alpha_soft * (0.5 - validity) * (1.0 - reliability)

    if dropout > 0.0:
        keep = torch.rand_like(validity) >= dropout
        hard = hard & keep
        soft = soft & keep

    masked = logits - torch.where(soft, penalty, torch.zeros_like(penalty))
    neg_inf = torch.finfo(masked.dtype).min
    masked = masked.masked_fill(hard, neg_inf)

    if feasible is not None:
        feasible = feasible.bool()
        masked = masked.masked_fill(~feasible, neg_inf)
        # If the learned mask removed every feasible action, fall back to the
        # feasibility mask so that the policy always has an action to take.
        empty = (masked <= neg_inf).all(dim=-1, keepdim=True)
        fallback = logits.masked_fill(~feasible, neg_inf)
        masked = torch.where(empty, fallback, masked)
    return masked


class PerformanceCurriculum:
    """Performance-driven curriculum for the masking parameters.

    The progress p starts at 0 and is increased by `step` when the average
    completion rate over the last `window` episodes is at least `threshold`,
    with at least `min_gap` episodes between two increases. The masking
    parameters are linear in p:

        epsilon(p)    = 0.3 - 0.25 p   (mask dropout rate during training)
        alpha_soft(p) = 1.0 + 4.0 p    (soft masking strength)
        tau_r(p)      = 0.6 + 0.3 p    (reliability threshold)
    """

    def __init__(
        self,
        window: int = 5,
        threshold: float = 0.95,
        step: float = 0.1,
        min_gap: int = 5,
        eps_range: Tuple[float, float] = (0.3, 0.05),
        alpha_range: Tuple[float, float] = (1.0, 5.0),
        tau_range: Tuple[float, float] = (0.6, 0.9),
    ):
        self.window, self.threshold, self.step, self.min_gap = window, threshold, step, min_gap
        self.eps_range, self.alpha_range, self.tau_range = eps_range, alpha_range, tau_range
        self.progress = 0.0
        self._recent = deque(maxlen=window)
        self._since_increase = 0

    def update(self, completion_rate: float) -> float:
        """Record the completion rate (fraction in [0, 1]) of a finished episode."""
        self._recent.append(completion_rate)
        self._since_increase += 1
        if (
            self.progress < 1.0
            and len(self._recent) == self.window
            and self._since_increase >= self.min_gap
            and sum(self._recent) / self.window >= self.threshold
        ):
            self.progress = min(1.0, round(self.progress + self.step, 10))
            self._since_increase = 0
        return self.progress

    @staticmethod
    def _lerp(r: Tuple[float, float], p: float) -> float:
        return r[0] + (r[1] - r[0]) * p

    def params(self, training: bool = True) -> Dict[str, float]:
        """Current masking parameters. At evaluation the final setting is used
        without mask dropout."""
        p = self.progress if training else 1.0
        return {
            "dropout": self._lerp(self.eps_range, p) if training else 0.0,
            "alpha_soft": self._lerp(self.alpha_range, p),
            "tau_r": self._lerp(self.tau_range, p),
        }


class DualHeadActor(nn.Module):
    """Shared state encoder with separate policy heads for two robots
    (e.g. the atmospheric and vacuum robots of a cluster tool)."""

    def __init__(self, state_dim: int, num_actions_a: int, num_actions_b: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.body, feat_dim = _mlp(state_dim, hidden)
        self.head_a = nn.Linear(feat_dim, num_actions_a)
        self.head_b = nn.Linear(feat_dim, num_actions_b)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(state)
        return self.head_a(h), self.head_b(h)

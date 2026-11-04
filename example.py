"""Minimal usage example of the SMART masking components on a toy problem.

A synthetic environment has 12 discrete actions. An action is physically
feasible with probability 0.8, and a feasible action is logically valid when the
corresponding state feature is positive. The VRNet learns this rule from data,
and its predictions are then used to mask the logits of a dual-head actor.
"""

import torch

from smart_masking import DualHeadActor, PerformanceCurriculum, VRNet, adaptive_mask, vrnet_loss

torch.manual_seed(0)
STATE_DIM, N_A, N_B = 16, 4, 8          # two robots with 4 and 8 actions
N = N_A + N_B


def sample_batch(batch_size=256):
    state = torch.randn(batch_size, STATE_DIM)
    feasible = torch.rand(batch_size, N) < 0.8
    valid = feasible & (state[:, :N] > 0)
    return state, feasible, valid


# 1) Train the VRNet on (state, validity label) pairs.
vrnet = VRNet(STATE_DIM, N)
opt = torch.optim.Adam(vrnet.parameters(), lr=1e-3)
for it in range(2000):
    state, feasible, valid = sample_batch()
    v, r = vrnet(state)
    loss = vrnet_loss(v, r, valid, lam=0.1, label_mask=feasible)
    opt.zero_grad()
    loss.backward()
    opt.step()

state, feasible, valid = sample_batch(4096)
with torch.no_grad():
    v, r = vrnet(state)
acc = (((v > 0.5) == valid) | ~feasible).float().mean().item()
print(f"VRNet validity accuracy on feasible actions: {acc:.3f}")

# 2) Performance-driven curriculum: completion rates of successive episodes.
curriculum = PerformanceCurriculum()
for ep, cr in enumerate([0.6, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * 50, start=1):
    p = curriculum.update(cr)
print(f"Curriculum progress after {ep} episodes: p = {p:.1f}, "
      + ", ".join(f"{k} = {x:.2f}" for k, x in curriculum.params().items()))

# 3) Mask the logits of a dual-head actor with the final curriculum setting.
actor = DualHeadActor(STATE_DIM, N_A, N_B)
prm = curriculum.params(training=False)
with torch.no_grad():
    logits_a, logits_b = actor(state)
    logits = torch.cat([logits_a, logits_b], dim=-1)
    masked = adaptive_mask(logits, v, r, prm["tau_r"], prm["alpha_soft"], feasible=feasible)
removed = masked <= torch.finfo(masked.dtype).min
invalid = feasible & ~valid
print(f"Invalid feasible actions hard-masked: {removed[invalid].float().mean().item():.3f}")
print(f"Valid actions kept: {(~removed[valid]).float().mean().item():.3f}")

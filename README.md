# SMART: Self-Tuning Action Masking with Adaptive Reliability Threshold

This repository provides an implementation of the masking components of SMART, proposed in
"Self-Tuning Action Masking with Adaptive Reliability Threshold (SMART) for Reinforcement Learning-Based Semiconductor Cluster Tool Scheduling".

## Contents

| File | Description |
|---|---|
| `smart_masking.py` | VRNet, adaptive masking (AMI), performance-driven curriculum, and dual-head actor |
| `example.py` | Minimal example on a synthetic constrained action space |
| `requirements.txt` | Dependencies |

The cluster tool simulator, the scenario settings, and the experiment scripts are not included because they are subject to an industrial collaboration agreement. The model and curriculum settings used in the paper are listed in the paper.

## Components

- **VRNet** (`VRNet`, `vrnet_loss`): for each action, predicts the probability that the action is valid and the reliability of this prediction. It is trained with `L = L_validity + lambda * L_reliability`, where the reliability target is 1 when the validity prediction is correct.
- **Adaptive masking** (`adaptive_mask`): an action predicted invalid with reliability of at least `tau_r` is hard-masked, an action predicted invalid with lower reliability receives the soft penalty `alpha_soft * (0.5 - v) * (1 - r)`, and an action predicted valid is not changed. Physically infeasible actions are always removed.
- **Performance-driven curriculum** (`PerformanceCurriculum`): the progress `p` increases by 0.1 when the average completion rate over the last five episodes is at least 95%, with at least five episodes between increases. The mask dropout rate, soft masking strength, and reliability threshold are linear in `p`.
- **Dual-head actor** (`DualHeadActor`): a shared encoder with one policy head per robot.

## Usage

The components can be combined with any discrete-action RL algorithm (the paper uses soft actor-critic). The environment needs to provide a mask of physically feasible actions, validity labels for training the VRNet, and the completion rate of each episode.

```python
from smart_masking import VRNet, vrnet_loss, adaptive_mask, PerformanceCurriculum

vrnet = VRNet(state_dim, num_actions)
curriculum = PerformanceCurriculum()

# action selection
v, r = vrnet(state)
prm = curriculum.params(training=True)
masked_logits = adaptive_mask(logits, v, r, prm["tau_r"], prm["alpha_soft"],
                              feasible=feasible_mask, dropout=prm["dropout"])

# VRNet update (lambda = 0.1 in the paper)
loss = vrnet_loss(v, r, validity_label, lam=0.1, label_mask=feasible_mask)

# end of an episode
curriculum.update(completion_rate)
```

Run the example:

```bash
pip install -r requirements.txt
python example.py
```

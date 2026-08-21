# === banglaHate/src/losses.py ===
"""
Custom loss functions for Consistency-Constrained Multi-Task Learning.

Contains:
  1. FocalLoss — class-imbalance-aware cross-entropy (Lin et al., 2017)
  2. ConsistencyPenaltyLoss — soft differentiable penalty for logically
     contradictory multi-task predictions based on BanglaMultiHate annotation rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance (Lin et al., ICCV 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (Tensor, optional): Per-class weights. Shape: (num_classes,).
        gamma (float): Focusing parameter. Higher gamma → more focus on hard examples.
                       Default: 2.0.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        Returns:
            Scalar focal loss value
        """
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)  # probability of correct class
        focal_weight = ((1 - pt) ** self.gamma)
        return (focal_weight * ce).mean()


class ConsistencyPenaltyLoss(nn.Module):
    """
    Differentiable soft constraint loss penalizing logically contradictory
    multi-task predictions.

    Based on BanglaMultiHate annotation schema rules:
      Rule 1: If hate_type = "None" → target MUST be "None"
                                   AND severity MUST be "Little to None"
      Rule 2 (contrapositive): If target ≠ "None" OR severity ≠ "Little to None"
                               → hate_type MUST NOT be "None"

    The loss computes soft products of contradictory probability masses:
      v1 = P(type=None) × P(target≠None)
      v2 = P(type=None) × P(sev≠Little)
      v3 = P(type=None) × P(sev=Severe)   [extra penalty for worst violation]

    L_consist = mean(v1 + v2 + 2*v3)

    This is fully differentiable and allows gradient backpropagation.

    Args:
        type_none_idx (int): Index of "None" in type_of_hate label list. Default: 5.
        target_none_idx (int): Index of "None" in target_of_hate label list. Default: 0.
        sev_little_idx (int): Index of "Little to None" in severity label list. Default: 0.
        sev_severe_idx (int): Index of "Severe" in severity label list. Default: 2.
    """

    def __init__(self, type_none_idx=5, target_none_idx=0,
                 sev_little_idx=0, sev_severe_idx=2):
        super().__init__()
        self.type_none_idx = type_none_idx
        self.target_none_idx = target_none_idx
        self.sev_little_idx = sev_little_idx
        self.sev_severe_idx = sev_severe_idx

    def forward(self, type_logits, target_logits, severity_logits):
        """
        Args:
            type_logits: Shape (batch_size, 6) — raw logits for hate type
            target_logits: Shape (batch_size, 5) — raw logits for target
            severity_logits: Shape (batch_size, 3) — raw logits for severity
        Returns:
            Scalar consistency penalty loss
        """
        # Convert logits to soft probabilities
        p_type = F.softmax(type_logits, dim=-1)
        p_target = F.softmax(target_logits, dim=-1)
        p_sev = F.softmax(severity_logits, dim=-1)

        # Extract relevant probability masses
        p_type_none = p_type[:, self.type_none_idx]           # P(type = None)
        p_target_has = 1.0 - p_target[:, self.target_none_idx]  # P(target ≠ None)
        p_sev_has = 1.0 - p_sev[:, self.sev_little_idx]        # P(sev ≠ Little)
        p_sev_severe = p_sev[:, self.sev_severe_idx]            # P(sev = Severe)

        # Violation terms (soft products — should be zero if logically consistent)
        v1 = p_type_none * p_target_has    # type=None but target≠None
        v2 = p_type_none * p_sev_has       # type=None but severity≠Little
        v3 = p_type_none * p_sev_severe    # type=None but severity=Severe (worst)

        # Aggregate: extra penalty weight on severe violations
        penalty = (v1 + v2 + 2.0 * v3).mean()

        return penalty

"""Loss function for icon/object detection CNN."""

import torch
import torch.nn as nn

from sbg.models.icon_net import TARGETS


def icon_loss(pred: dict[str, torch.Tensor],
              targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """Combined loss for all detection heads.

    For each target:
    - BCE loss on presence logit (always)
    - Smooth L1 loss on (x, y) coordinates (only when target is present)
    """
    total = torch.tensor(0.0, device=next(iter(pred.values())).device)

    for key in TARGETS:
        p = pred[key]
        t = targets[key]
        present = t[:, 0]  # (B,)

        # Presence loss (BCE with logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            p[:, 0], present, reduction="mean"
        )
        total = total + bce

        # Coordinate loss (only for present targets)
        mask = present.bool()
        if mask.any():
            coord_pred = p[mask, 1:]      # (N, 2)
            coord_target = t[mask, 1:]    # (N, 2)
            coord_loss = nn.functional.smooth_l1_loss(coord_pred, coord_target)
            total = total + coord_loss * 5.0  # weight coords higher

    return total

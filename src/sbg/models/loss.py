"""Loss function for icon detection CNN."""

import torch
import torch.nn as nn


def icon_loss(pred: dict[str, torch.Tensor],
              ball_target: torch.Tensor,
              pin_target: torch.Tensor) -> torch.Tensor:
    """Combined loss for both icon heads.

    For each icon:
    - BCE loss on presence logit (always)
    - Smooth L1 loss on (x, y) coordinates (only when icon is present)
    """
    total = torch.tensor(0.0, device=pred["ball"].device)

    for key, target in [("ball", ball_target), ("pin", pin_target)]:
        p = pred[key]
        present = target[:, 0]  # (B,)

        # Presence loss (BCE with logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            p[:, 0], present, reduction="mean"
        )
        total = total + bce

        # Coordinate loss (only for present icons)
        mask = present.bool()
        if mask.any():
            coord_pred = p[mask, 1:]      # (N, 2)
            coord_target = target[mask, 1:]  # (N, 2)
            coord_loss = nn.functional.smooth_l1_loss(coord_pred, coord_target)
            total = total + coord_loss * 5.0  # weight coords higher

    return total

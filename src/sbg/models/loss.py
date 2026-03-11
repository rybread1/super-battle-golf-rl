"""Loss function for icon/object detection CNN."""

import torch
import torch.nn as nn

from sbg.models.icon_net import TARGETS


def _make_gaussian_heatmap(target: torch.Tensor, H: int, W: int,
                           sigma: float = 2.0) -> torch.Tensor:
    """Create a gaussian heatmap target from normalized (x, y) coordinates.

    target: (B, 3) — [present, x_norm, y_norm]
    Returns: (B, 1, H, W) heatmap with gaussian blob at target location,
             or zeros if target is not present.
    """
    B = target.shape[0]
    device = target.device

    # Grid of cell centers in [0, 1]
    gy = ((torch.arange(H, device=device).float() + 0.5) / H).view(1, 1, H, 1)
    gx = ((torch.arange(W, device=device).float() + 0.5) / W).view(1, 1, 1, W)

    # Target coords (B, 1, 1, 1)
    tx = target[:, 1].view(B, 1, 1, 1)
    ty = target[:, 2].view(B, 1, 1, 1)

    # Sigma in normalized coords (sigma cells / grid size)
    sx = sigma / W
    sy = sigma / H

    # Gaussian blob
    heatmap = torch.exp(-((gx - tx) ** 2 / (2 * sx ** 2) + (gy - ty) ** 2 / (2 * sy ** 2)))

    # Zero out heatmaps where target is not present
    present = target[:, 0].view(B, 1, 1, 1)
    heatmap = heatmap * present

    return heatmap  # (B, 1, H, W)


def icon_loss(pred: dict[str, torch.Tensor],
              targets: dict[str, torch.Tensor],
              heatmaps: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """Combined loss for all detection heads.

    For each target:
    - BCE loss on presence logit (always)
    - Smooth L1 loss on (x, y) coordinates, weighted by presence value
      (supports soft labels from mixup)
    - Heatmap supervision loss (MSE vs gaussian blob, if heatmaps provided)
    """
    total = torch.tensor(0.0, device=next(iter(pred.values())).device)

    for key in TARGETS:
        p = pred[key]
        t = targets[key]
        present = t[:, 0]  # (B,) — 0/1 or soft value from mixup

        # Presence loss (BCE with logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            p[:, 0], present, reduction="mean"
        )
        total = total + bce

        # Coordinate loss, weighted by presence (soft-compatible)
        if present.sum() > 0:
            coord_pred = p[:, 1:]      # (B, 2)
            coord_target = t[:, 1:]    # (B, 2)
            per_sample = nn.functional.smooth_l1_loss(
                coord_pred, coord_target, reduction="none"
            ).mean(dim=1)  # (B,)
            coord_loss = (per_sample * present).sum() / present.sum()
            total = total + coord_loss * 10.0

        # Heatmap supervision loss
        if heatmaps is not None and key in heatmaps:
            raw_heatmap = heatmaps[key]  # (B, 1, H, W) — raw logits from head
            _, _, H, W = raw_heatmap.shape
            gt_heatmap = _make_gaussian_heatmap(t, H, W)
            # Sigmoid on raw heatmap to get [0, 1], then MSE vs gaussian target
            hm_loss = nn.functional.mse_loss(torch.sigmoid(raw_heatmap), gt_heatmap)
            total = total + hm_loss * 5.0

    return total

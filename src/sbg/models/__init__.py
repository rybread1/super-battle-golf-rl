"""Learned models for game element detection."""

from sbg.models.icon_net import IconNet
from sbg.models.loss import icon_loss

__all__ = ["IconNet", "icon_loss"]

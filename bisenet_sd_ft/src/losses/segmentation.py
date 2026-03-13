from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_index: int,
    ignore_index: int = 255,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, class_index]
    valid_mask = (target != ignore_index).float()
    target_mask = (target == class_index).float()
    numerator = 2.0 * torch.sum(probs * target_mask * valid_mask)
    denominator = torch.sum((probs + target_mask) * valid_mask) + eps
    return 1.0 - numerator / denominator


def boundary_region(mask: torch.Tensor, width: int = 4) -> torch.Tensor:
    hair_mask = mask.float().unsqueeze(1)
    kernel = width * 2 + 1
    dilated = F.max_pool2d(hair_mask, kernel_size=kernel, stride=1, padding=width)
    eroded = -F.max_pool2d(-hair_mask, kernel_size=kernel, stride=1, padding=width)
    return (dilated - eroded > 0).float().squeeze(1)


class HairSegLoss(nn.Module):
    def __init__(self, cfg: dict[str, Any], num_classes: int = 16) -> None:
        super().__init__()
        self.ignore_index = int(cfg["data"]["coarse_labels"]["ignore"])
        self.face_index = int(cfg["data"]["coarse_labels"]["face"])
        self.hair_index = int(cfg["data"]["coarse_labels"]["hair"])
        self.aux_ce_weight = float(cfg["loss"].get("aux_ce_weight", 0.4))
        self.ce_weight = float(cfg["loss"].get("ce_weight", 1.0))
        self.hair_dice_weight = float(cfg["loss"].get("hair_dice_weight", 0.5))
        self.face_dice_weight = float(cfg["loss"].get("face_dice_weight", 0.2))
        self.boundary_weight = float(cfg["loss"].get("boundary_weight", 0.2))
        self.boundary_width = int(cfg["loss"].get("boundary_width", 4))

        class_weights = torch.ones(num_classes, dtype=torch.float32)
        class_weights[self.face_index] = float(cfg["loss"]["class_weights"].get("face", 2.0))
        class_weights[self.hair_index] = float(cfg["loss"]["class_weights"].get("hair", 4.0))
        class_weights[0] = float(cfg["loss"]["class_weights"].get("background", 1.0))
        self.register_buffer("class_weights", class_weights)

    def _cross_entropy(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=self.ignore_index)

    def _boundary_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hair_prob = torch.softmax(logits, dim=1)[:, self.hair_index]
        gt_hair = (target == self.hair_index).float()
        valid = (target != self.ignore_index).float()
        boundary = boundary_region(gt_hair, width=self.boundary_width) * valid
        if torch.count_nonzero(boundary) == 0:
            return logits.new_tensor(0.0)
        boundary_target = gt_hair[boundary > 0]
        boundary_prob = hair_prob[boundary > 0].clamp(1.0e-5, 1.0 - 1.0e-5)
        return F.binary_cross_entropy(boundary_prob, boundary_target)

    def forward(self, logits: torch.Tensor, logits16: torch.Tensor, logits32: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        ce_main = self._cross_entropy(logits, target)
        ce_aux_16 = self._cross_entropy(logits16, target)
        ce_aux_32 = self._cross_entropy(logits32, target)
        hair_dice = binary_dice_loss(logits, target, class_index=self.hair_index, ignore_index=self.ignore_index)
        face_dice = binary_dice_loss(logits, target, class_index=self.face_index, ignore_index=self.ignore_index)
        boundary = self._boundary_loss(logits, target)
        total = (
            self.ce_weight * ce_main
            + self.aux_ce_weight * (ce_aux_16 + ce_aux_32)
            + self.hair_dice_weight * hair_dice
            + self.face_dice_weight * face_dice
            + self.boundary_weight * boundary
        )
        return {
            "total": total,
            "ce_main": ce_main.detach(),
            "ce_aux_16": ce_aux_16.detach(),
            "ce_aux_32": ce_aux_32.detach(),
            "hair_dice_loss": hair_dice.detach(),
            "face_dice_loss": face_dice.detach(),
            "boundary_loss": boundary.detach(),
        }

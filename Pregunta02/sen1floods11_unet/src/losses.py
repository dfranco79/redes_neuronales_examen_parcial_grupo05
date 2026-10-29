"""Pérdidas enmascaradas: solo cuentan los píxeles con valid=True."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCE(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pw = pos_weight

    def forward(self, logits, lbl, valid):
        logits, m = logits.squeeze(1).float(), valid.float()
        pw = torch.tensor(self.pw, device=logits.device) if self.pw else None
        loss = F.binary_cross_entropy_with_logits(logits, (lbl == 1).float(), pos_weight=pw, reduction="none")
        return (loss * m).sum() / m.sum().clamp_min(1.0)


class MaskedDice(nn.Module):
    """1 - Dice de la clase agua, calculado por imagen y promediado en el batch."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, lbl, valid):
        m = valid.float()
        p = torch.sigmoid(logits.squeeze(1).float()) * m
        t = (lbl == 1).float() * m
        dice = (2 * (p * t).sum((1, 2)) + self.smooth) / (p.sum((1, 2)) + t.sum((1, 2)) + self.smooth)
        return 1 - dice.mean()


class BCEDice(nn.Module):
    def __init__(self, pos_weight=3.0, lambda_dice=1.0):
        super().__init__()
        self.bce, self.dice, self.lam = MaskedBCE(pos_weight), MaskedDice(), lambda_dice

    def forward(self, logits, lbl, valid):
        return self.bce(logits, lbl, valid) + self.lam * self.dice(logits, lbl, valid)

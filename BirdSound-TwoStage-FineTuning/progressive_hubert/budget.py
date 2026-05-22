"""Matched optimizer-step budgets across training strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class TrainingBudget:
    steps_per_epoch_single: int
    steps_per_epoch_overlap: int
    phase1_epochs: int
    phase2_epochs: int

    @property
    def progressive_total_steps(self) -> int:
        return (
            self.phase1_epochs * self.steps_per_epoch_single
            + self.phase2_epochs * self.steps_per_epoch_overlap
        )

    def phase_steps_for_progressive(
        self, match_progressive_total: bool, phase1_fraction: float
    ) -> tuple[int | None, int | None]:
        if not match_progressive_total:
            return (
                self.phase1_epochs * self.steps_per_epoch_single,
                self.phase2_epochs * self.steps_per_epoch_overlap,
            )
        total = self.progressive_total_steps
        p1 = int(total * phase1_fraction)
        return p1, total - p1

    def summary(self, match_progressive_total: bool, phase1_fraction: float) -> dict:
        p1, p2 = self.phase_steps_for_progressive(match_progressive_total, phase1_fraction)
        return {
            "progressive_total_steps": self.progressive_total_steps,
            "progressive_phase1_steps": p1,
            "progressive_phase2_steps": p2,
            "match_progressive_total": match_progressive_total,
        }


def build_budget(
    single_loader: DataLoader,
    overlap_loader: DataLoader,
    phase1_epochs: int,
    phase2_epochs: int,
) -> TrainingBudget:
    return TrainingBudget(
        steps_per_epoch_single=len(single_loader),
        steps_per_epoch_overlap=len(overlap_loader),
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
    )

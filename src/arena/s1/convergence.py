"""Validation-driven learning-rate reductions and stopping for policy distillation."""
from dataclasses import asdict, dataclass
import math


@dataclass
class ConvergenceControl:
    learning_rate: float
    min_learning_rate: float
    min_epochs: float = 2.0
    min_delta: float = 0.002
    patience: int = 4
    reference_loss: float = math.inf
    best_loss: float = math.inf
    best_step: int = 0
    plateau_checks: int = 0
    floor_checks: int = 0
    reductions: int = 0

    def observe(self, loss: float, epoch: float, step: int) -> dict:
        if not math.isfinite(loss):
            raise ValueError("Validation loss must be finite")
        best = loss < self.best_loss
        if best:
            self.best_loss, self.best_step = loss, step
        improved = loss < self.reference_loss * (1.0 - self.min_delta)
        at_floor = self.learning_rate <= self.min_learning_rate * (1 + 1e-9)
        if improved:
            self.reference_loss = loss
            self.plateau_checks = self.floor_checks = 0
        else:
            self.plateau_checks += 1
            self.floor_checks = self.floor_checks + 1 if at_floor else 0
        reduced = False
        if self.plateau_checks >= 2 and not at_floor:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * 0.5)
            self.plateau_checks = 0
            self.floor_checks = 0
            self.reductions += 1
            reduced = True
        converged = epoch >= self.min_epochs and self.floor_checks >= self.patience
        return {"new_best": best, "meaningful_improvement": improved, "lr_reduced": reduced,
                "converged": converged, **asdict(self)}

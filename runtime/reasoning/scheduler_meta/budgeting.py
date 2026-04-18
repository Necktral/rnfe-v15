"""Budgeting determinista para META."""

from __future__ import annotations

from typing import Dict
from runtime.reasoning.scheduler_meta.context_features import normalize_feature_vector


CORE_FAMILY_COUNT = 6


def _reserved_optional_slots(features: Dict[str, float]) -> int:
    slots = 0
    if features["edge_pressure"] >= 0.7:
        slots += 1
    if features["contradiction_signal"] >= 0.45:
        slots += 2
    if (
        features["symbolic_regularity"] >= 0.4
        or features["law_fit_signal"] >= 0.4
    ):
        slots += 1
    return slots


def compute_budget(features: Dict[str, float], *, max_steps_override: int | None = None) -> Dict[str, float]:
    normalized = normalize_feature_vector(features)
    base_steps = 6
    dynamic_bonus = 0
    if normalized["uncertainty"] >= 0.6:
        dynamic_bonus += 1
    if normalized["contradiction_signal"] >= 0.5:
        dynamic_bonus += 1
    if normalized["causal_risk"] >= 0.5:
        dynamic_bonus += 1
    if normalized["edge_pressure"] >= 0.8:
        dynamic_bonus -= 1
    max_steps = base_steps + dynamic_bonus
    required_min_steps = CORE_FAMILY_COUNT + _reserved_optional_slots(normalized)
    max_steps = max(required_min_steps, max_steps)
    max_steps = max(CORE_FAMILY_COUNT, min(10, max_steps))
    if max_steps_override is not None:
        max_steps = max(CORE_FAMILY_COUNT, min(10, int(max_steps_override)))
        max_steps = max(max_steps, required_min_steps)

    risk_budget = min(
        1.0,
        0.3
        + (0.4 * normalized["uncertainty"])
        + (0.2 * normalized["contradiction_signal"])
        + (0.1 * normalized["causal_risk"]),
    )
    return {
        "max_steps": float(max_steps),
        "risk_budget": risk_budget,
        "cost_budget": float(max_steps),
    }

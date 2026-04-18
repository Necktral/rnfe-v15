"""Extracción determinista y normalización de features para META."""

from __future__ import annotations

from typing import Any, Dict, Mapping


DEFAULT_FEATURES: Dict[str, float] = {
    "uncertainty": 0.25,
    "contradiction_signal": 0.0,
    "continuity_recent": 1.0,
    "edge_pressure": 0.0,
    "causal_risk": 0.0,
    "symbolic_regularity": 0.0,
    "law_fit_signal": 0.0,
}


def _clamp_01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def normalize_feature_vector(features: Mapping[str, Any] | None) -> Dict[str, float]:
    """Normaliza un vector parcial de features con defaults y clamps [0, 1]."""
    source: Mapping[str, Any] = features or {}
    uncertainty = _as_float(source.get("uncertainty"), default=DEFAULT_FEATURES["uncertainty"])
    contradiction = _as_float(
        source.get("contradiction_signal"),
        default=DEFAULT_FEATURES["contradiction_signal"],
    )
    continuity_recent = _as_float(
        source.get("continuity_recent"),
        default=DEFAULT_FEATURES["continuity_recent"],
    )
    edge_pressure = _as_float(source.get("edge_pressure"), default=DEFAULT_FEATURES["edge_pressure"])
    symbolic_regularity = _as_float(
        source.get("symbolic_regularity"),
        default=DEFAULT_FEATURES["symbolic_regularity"],
    )
    law_fit_signal = _as_float(
        source.get("law_fit_signal"),
        default=DEFAULT_FEATURES["law_fit_signal"],
    )
    explicit_causal_risk = source.get("causal_risk")
    if isinstance(explicit_causal_risk, (int, float)):
        causal_risk = float(explicit_causal_risk)
    else:
        counterfactual_gap = _as_float(source.get("counterfactual_gap"), default=0.0)
        causal_risk = abs(counterfactual_gap)

    return {
        "uncertainty": _clamp_01(uncertainty),
        "contradiction_signal": _clamp_01(contradiction),
        "continuity_recent": _clamp_01(continuity_recent),
        "edge_pressure": _clamp_01(edge_pressure),
        "causal_risk": _clamp_01(causal_risk),
        "symbolic_regularity": _clamp_01(symbolic_regularity),
        "law_fit_signal": _clamp_01(law_fit_signal),
    }


def extract_context_features(context: Dict[str, Any]) -> Dict[str, float]:
    features = normalize_feature_vector(context)
    observation = context.get("observation", {})
    if isinstance(observation, dict) and observation.get("alarm") is True:
        features["contradiction_signal"] = max(features["contradiction_signal"], 0.4)
        features["edge_pressure"] = max(features["edge_pressure"], 0.3)
    return features

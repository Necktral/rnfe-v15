"""Política adaptativa y explicable para selección de familias."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple
from runtime.reasoning.scheduler_meta.context_features import normalize_feature_vector


FAMILY_POOL = [
    "abd",
    "ana",
    "cau",
    "ctf",
    "ded",
    "prob",
    "dia_adv",
    "heur",
    "fal_guard",
    "eml_sr",
]
CORE_SEQUENCE = ["abd", "ana", "cau", "ctf", "ded", "prob"]


def score_families(features: Dict[str, float]) -> Dict[str, float]:
    scores = {family: 0.1 for family in FAMILY_POOL}
    scores["abd"] += 0.3
    scores["ana"] += 0.2 + (0.2 * features["uncertainty"])
    scores["cau"] += 0.2 + (0.2 * features["causal_risk"])
    scores["ctf"] += 0.2 + (0.2 * features["causal_risk"])
    scores["ded"] += 0.2 + (0.2 * (1.0 - features["continuity_recent"]))
    scores["prob"] += 0.2 + (0.25 * features["uncertainty"])
    scores["heur"] += 0.1 + (0.35 * features["edge_pressure"])
    scores["dia_adv"] += 0.1 + (0.35 * features["contradiction_signal"])
    scores["fal_guard"] += 0.1 + (0.25 * features["contradiction_signal"])
    scores["eml_sr"] += (
        0.1
        + (0.3 * features.get("symbolic_regularity", 0.0))
        + (0.25 * features.get("law_fit_signal", 0.0))
    )
    return scores


def _dedup(sequence: List[str]) -> List[str]:
    out: List[str] = []
    for family in sequence:
        if family not in out:
            out.append(family)
    return out


def _active_optional_families(
    features: Dict[str, float], *, allow_experimental: bool
) -> List[str]:
    active: List[str] = []
    # Gate formal en 0.7 + banda muerta para reducir oscilaciones en torno al umbral.
    heur_activation = (
        features["edge_pressure"] >= 0.7
        and (
            features["edge_pressure"] >= 0.715
            or features["uncertainty"] >= 0.6
        )
    )
    if heur_activation:
        active.append("heur")
    if features["contradiction_signal"] >= 0.45:
        active.extend(["dia_adv", "fal_guard"])
    if allow_experimental and (
        features["symbolic_regularity"] >= 0.4
        or features["law_fit_signal"] >= 0.4
    ):
        active.append("eml_sr")
    return active


def select_sequence(
    *,
    features: Dict[str, float],
    budget: Dict[str, float],
    allow_experimental: bool = False,
) -> Tuple[List[str], Dict[str, float], str]:
    normalized = normalize_feature_vector(features)
    scores = score_families(normalized)
    requested_steps = int(float(budget.get("max_steps", len(CORE_SEQUENCE))))
    max_steps = max(len(CORE_SEQUENCE), min(10, requested_steps))
    active_optional = _active_optional_families(
        normalized, allow_experimental=allow_experimental
    )

    optional_slots = max(0, max_steps - len(CORE_SEQUENCE))
    optional_selected = active_optional[:optional_slots]
    sequence: List[str] = ["abd", *optional_selected, *CORE_SEQUENCE[1:]]
    sequence = _dedup(sequence)

    remaining = [fam for fam in FAMILY_POOL if fam not in sequence]
    if remaining:
        recommended_next = sorted(remaining, key=lambda fam: (-scores[fam], fam))[0]
    else:
        recommended_next = "prob"
    return sequence, scores, recommended_next


def is_eml_experimental_enabled() -> bool:
    mode = os.environ.get("RNFE_EML_MODE", "disabled").strip().lower()
    if mode != "shadow":
        return False
    allowlist = os.environ.get("RNFE_META_EXPERIMENTAL_FAMILIES", "")
    enabled = {item.strip().lower() for item in allowlist.split(",") if item.strip()}
    return "eml_sr" in enabled

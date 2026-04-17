"""Scheduler META para seleccionar y secuenciar familias de razonamiento."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, List, Any

from runtime.reasoning.contracts import ReasoningTraceStep


class MetaScheduler:
    """Implementación mínima y trazable del scheduler META."""

    DEFAULT_SEQUENCE = ["abd", "ana", "cau", "ctf", "ded", "prob"]

    def __init__(self, sequence: List[str] | None = None):
        self.sequence = sequence or list(self.DEFAULT_SEQUENCE)

    def run(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(context or {})
        traces: List[ReasoningTraceStep] = []
        for family in self.sequence:
            module = import_module(f"runtime.reasoning.families.{family}")
            result = module.execute(state)
            state.update(result.get("state_delta", {}))
            traces.append(
                ReasoningTraceStep(
                    family=family.upper(),
                    status=result.get("status", "ok"),
                    detail=result,
                )
            )
        return {
            "meta_family": "META",
            "sequence": [f.upper() for f in self.sequence],
            "trace": [t.__dict__ for t in traces],
            "state": state,
        }


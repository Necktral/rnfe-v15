"""Escenario térmico espacial 5x5 — primer mundo intermedio con grilla real.

Este escenario modela una malla de 5x5 = 25 celdas, cada una con temperatura
propia y estado de enfriamiento local.  A diferencia de los escenarios
escalares (1x1), la grilla introduce:

* **Difusión térmica** entre vecinos (Von-Neumann 4-vecinos).
* **Fuentes de calor localizadas** (``heat_sources``).
* **Intervención global** que activa/desactiva enfriamiento en *todas* las celdas.
* **Agregados globales** (mean_temp, max_temp, hotspot_count, hotspot_fraction)
  que permiten al scheduler razonar sin inspeccionar cada celda.
* **``world_level``** derivado del agregado ``hotspot_fraction``, no manual.

El estado completo es determinista y reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .compatibility import ScenarioStructuralProfile
from .causal_signature import (
    CausalEdge,
    InterventionEffect,
    ScenarioCausalSignature,
)
from .scenario import (
    CognitiveScenario,
    ScenarioConfig,
    ScenarioObservation,
    ScenarioTransition,
)


# ─── Grid constants ──────────────────────────────────────────────────────────

GRID_ROWS: int = 5
GRID_COLS: int = 5
CELL_COUNT: int = GRID_ROWS * GRID_COLS


# ─── Grid state ──────────────────────────────────────────────────────────────

@dataclass
class ThermalGridState:
    """Estado interno de la grilla térmica 5x5.

    Attributes:
        temperatures: Lista plana de 25 floats (row-major) en [0.0, 1.0].
        cooling_active: Lista plana de 25 bools (enfriamiento por celda).
        global_cooling: Si la intervención de enfriamiento global está activa.
    """

    temperatures: List[float]
    cooling_active: List[bool]
    global_cooling: bool


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _neighbors(idx: int, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> List[int]:
    """Retorna índices Von-Neumann (4-vecinos) de la celda ``idx``."""
    r, c = divmod(idx, cols)
    result: List[int] = []
    if r > 0:
        result.append((r - 1) * cols + c)
    if r < rows - 1:
        result.append((r + 1) * cols + c)
    if c > 0:
        result.append(r * cols + (c - 1))
    if c < cols - 1:
        result.append(r * cols + (c + 1))
    return result


def _compute_aggregates(
    temperatures: List[float],
    alarm_threshold: float,
) -> Dict[str, float]:
    """Computa agregados globales del mundo.

    Returns:
        Dict con mean_temp, max_temp, min_temp, hotspot_count, hotspot_fraction.
    """
    n = len(temperatures)
    mean_temp = sum(temperatures) / n
    max_temp = max(temperatures)
    min_temp = min(temperatures)
    hotspot_count = sum(1 for t in temperatures if t >= alarm_threshold)
    hotspot_fraction = hotspot_count / n
    return {
        "mean_temp": round(mean_temp, 6),
        "max_temp": round(max_temp, 6),
        "min_temp": round(min_temp, 6),
        "hotspot_count": hotspot_count,
        "hotspot_fraction": round(hotspot_fraction, 6),
    }


# ─── Scenario ────────────────────────────────────────────────────────────────

class ThermalGrid5x5Scenario(CognitiveScenario):
    """Escenario térmico espacial 5x5 con difusión, agregados y trazabilidad.

    Niveles del mundo derivados de ``hotspot_fraction``:
    - Nivel 1 (NORMAL):   hotspot_fraction < warning_fraction
    - Nivel 2 (WARNING):  warning_fraction ≤ hotspot_fraction < critical_fraction
    - Nivel 3 (CRITICAL): hotspot_fraction ≥ critical_fraction

    Parámetros default:
    - ``warning_fraction=0.20`` → ≥5 hotspots de 25 activa nivel 2
    - ``critical_fraction=0.40`` → ≥10 hotspots de 25 activa nivel 3
    """

    def __init__(
        self,
        *,
        initial_temperatures: Optional[List[float]] = None,
        alarm_threshold: float = 0.85,
        cooling_effect: float = 0.07,
        diffusion_rate: float = 0.02,
        warning_fraction: float = 0.20,
        critical_fraction: float = 0.40,
        heat_sources: Optional[List[int]] = None,
        heat_source_intensity: float = 0.03,
    ):
        """Inicializa grilla térmica 5x5.

        Args:
            initial_temperatures: Lista de 25 temperaturas iniciales.
                Si ``None``, se genera una distribución determinista por defecto.
            alarm_threshold: Umbral de alarma por celda.
            cooling_effect: Reducción de temperatura por paso cuando cooling activo.
            diffusion_rate: Tasa de difusión térmica entre vecinos por paso.
            warning_fraction: Fracción de hotspots para nivel 2.
            critical_fraction: Fracción de hotspots para nivel 3.
            heat_sources: Índices de celdas con fuentes de calor fijas.
                Si ``None``, se usan las esquinas y el centro [0, 4, 12, 20, 24].
            heat_source_intensity: Calor adicional por paso en celdas fuente.
        """
        self._alarm_threshold = alarm_threshold
        self._cooling_effect = cooling_effect
        self._diffusion_rate = diffusion_rate
        self._warning_fraction = warning_fraction
        self._critical_fraction = critical_fraction
        self._heat_sources = heat_sources if heat_sources is not None else [0, 4, 12, 20, 24]
        self._heat_source_intensity = heat_source_intensity

        if initial_temperatures is not None:
            if len(initial_temperatures) != CELL_COUNT:
                raise ValueError(
                    f"initial_temperatures debe tener {CELL_COUNT} elementos, "
                    f"recibió {len(initial_temperatures)}"
                )
            temps = list(initial_temperatures)
        else:
            # Distribución determinista: gradiente radial desde el centro
            center_r, center_c = GRID_ROWS // 2, GRID_COLS // 2
            max_dist = ((GRID_ROWS - 1) ** 2 + (GRID_COLS - 1) ** 2) ** 0.5
            temps = []
            for idx in range(CELL_COUNT):
                r, c = divmod(idx, GRID_COLS)
                dist = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
                # Centro caliente (~0.88), bordes más fríos (~0.65)
                t = 0.88 - 0.23 * (dist / max_dist)
                temps.append(round(t, 4))

        self._state = ThermalGridState(
            temperatures=temps,
            cooling_active=[False] * CELL_COUNT,
            global_cooling=False,
        )

        self._config = ScenarioConfig(
            name="thermal_grid_5x5",
            description=(
                "Grilla térmica 5x5 con difusión, fuentes de calor y "
                "enfriamiento global. World_level derivado de hotspot_fraction."
            ),
            main_variable="mean_temp",
            alarm_threshold=alarm_threshold,
            warning_threshold=warning_fraction,  # repurposed as warning_fraction
            interventions=["activate_cooling", "deactivate_cooling"],
            formula_template="GRID_CRITICAL -> ACTIVATE_COOLING",
            type_context={
                "GRID_NORMAL": "bool",
                "GRID_WARNING": "bool",
                "GRID_CRITICAL": "bool",
                "ACTIVATE_COOLING": "bool",
                "KEEP_IDLE": "bool",
            },
            world_shape=(GRID_ROWS, GRID_COLS),
        )

        # Precompute neighbor lists
        self._neighbor_cache: List[List[int]] = [
            _neighbors(i) for i in range(CELL_COUNT)
        ]

    # ─── Properties ──────────────────────────────────────────────────────

    @property
    def config(self) -> ScenarioConfig:
        return self._config

    @property
    def alarm_threshold(self) -> float:
        return self._alarm_threshold

    @property
    def warning_fraction(self) -> float:
        return self._warning_fraction

    @property
    def critical_fraction(self) -> float:
        return self._critical_fraction

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (GRID_ROWS, GRID_COLS)

    @property
    def cell_count(self) -> int:
        return CELL_COUNT

    @property
    def structural_profile(self) -> ScenarioStructuralProfile:
        cfg = self._config
        config_blob = json.dumps(
            {
                "name": cfg.name,
                "main_variable": cfg.main_variable,
                "alarm_threshold": cfg.alarm_threshold,
                "interventions": cfg.interventions,
                "formula_template": cfg.formula_template,
                "type_context": cfg.type_context,
                "world_shape": list(cfg.world_shape) if cfg.world_shape else None,
            },
            sort_keys=True,
        )
        config_hash = hashlib.sha256(config_blob.encode()).hexdigest()[:12]
        return ScenarioStructuralProfile(
            scenario_name=cfg.name,
            scenario_version="1.0",
            scenario_config_hash=config_hash,
            control_topology="threshold_single_loop",
            optimization_direction="minimize",
            intervention_semantics=tuple(cfg.interventions),
            counterfactual_policy="opposite_intervention",
            relation_polarity="lower_is_better",
            main_variable=cfg.main_variable,
        )

    @property
    def causal_signature(self) -> ScenarioCausalSignature:
        cfg = self._config
        return ScenarioCausalSignature(
            scenario_name=cfg.name,
            scenario_version="1.0",
            observable_variables=frozenset({
                "mean_temp", "max_temp", "hotspot_count",
                "hotspot_fraction", "global_cooling",
            }),
            control_variables=frozenset({"global_cooling"}),
            main_variable="mean_temp",
            optimization_direction="minimize",
            causal_polarity="lower_is_better",
            alarm_semantics="threshold_above",
            intervention_effects=(
                InterventionEffect(
                    intervention_name="activate_cooling",
                    target_variable="mean_temp",
                    expected_direction="-",
                    expected_magnitude=self._cooling_effect,
                    semantic_role="corrective",
                ),
                InterventionEffect(
                    intervention_name="deactivate_cooling",
                    target_variable="mean_temp",
                    expected_direction="+",
                    expected_magnitude=0.0,
                    semantic_role="neutral",
                ),
            ),
            counterfactual_policy="opposite_intervention",
            counterfactual_variable="mean_temp",
            causal_edges=(
                CausalEdge(source="external_heat", target="mean_temp", polarity="+"),
                CausalEdge(source="diffusion", target="mean_temp", polarity="?"),
                CausalEdge(source="global_cooling", target="mean_temp", polarity="-"),
                CausalEdge(source="mean_temp", target="hotspot_fraction", polarity="+"),
            ),
            proposition_vocabulary=frozenset({
                "GRID_CRITICAL", "GRID_WARNING", "GRID_NORMAL",
                "COOLING_ACTIVE", "ACTIVATE_COOLING", "KEEP_IDLE",
            }),
            metadata={
                "world_shape": [GRID_ROWS, GRID_COLS],
                "cell_count": CELL_COUNT,
            },
        )

    # ─── Level computation ───────────────────────────────────────────────

    def _compute_level(self, hotspot_fraction: float) -> int:
        """Determina world_level desde la fracción de hotspots.

        Returns:
            1 (NORMAL), 2 (WARNING) o 3 (CRITICAL).
        """
        if hotspot_fraction >= self._critical_fraction:
            return 3
        if self._warning_fraction > 0.0 and hotspot_fraction >= self._warning_fraction:
            return 2
        return 1

    # ─── Observation ─────────────────────────────────────────────────────

    def observe(self) -> ScenarioObservation:
        agg = _compute_aggregates(self._state.temperatures, self._alarm_threshold)
        level = self._compute_level(agg["hotspot_fraction"])

        if level == 3:
            proposition = "GRID_CRITICAL"
        elif level == 2:
            proposition = "GRID_WARNING"
        else:
            proposition = "GRID_NORMAL"

        propositions = [proposition]
        if self._state.global_cooling:
            propositions.append("COOLING_ACTIVE")

        state_dict: Dict[str, Any] = {
            # Aggregate fields (what the scheduler sees)
            "mean_temp": agg["mean_temp"],
            "max_temp": agg["max_temp"],
            "min_temp": agg["min_temp"],
            "hotspot_count": agg["hotspot_count"],
            "hotspot_fraction": agg["hotspot_fraction"],
            "global_cooling": self._state.global_cooling,
            # Grid metadata
            "world_shape": [GRID_ROWS, GRID_COLS],
            "cell_count": CELL_COUNT,
            # Full grid (for traceability; serialized as flat list)
            "grid_temperatures": list(self._state.temperatures),
            "grid_cooling": list(self._state.cooling_active),
        }

        return ScenarioObservation(
            state=state_dict,
            propositions=propositions,
            alarm=level == 3,
            level=level,
        )

    # ─── Transition logic ────────────────────────────────────────────────

    def _compute_transition(
        self,
        state: ThermalGridState,
        *,
        intervention: str,
        external_input: float,
    ) -> ThermalGridState:
        """Computa transición de estado para toda la grilla.

        Steps:
        1. Aplica intervención (global cooling on/off).
        2. Aplica calor externo uniforme a todas las celdas.
        3. Aplica calor extra en fuentes de calor.
        4. Aplica enfriamiento en celdas con cooling activo.
        5. Aplica difusión térmica entre vecinos.
        6. Clampea a [0.0, 1.0].
        """
        global_cooling = state.global_cooling
        if intervention == "activate_cooling":
            global_cooling = True
        elif intervention == "deactivate_cooling":
            global_cooling = False

        # Cooling per cell follows global state
        cooling = [global_cooling] * CELL_COUNT

        # Start from current temperatures
        new_temps = list(state.temperatures)

        # Step 2: External heat (uniform)
        for i in range(CELL_COUNT):
            new_temps[i] += external_input

        # Step 3: Heat sources
        for src_idx in self._heat_sources:
            if 0 <= src_idx < CELL_COUNT:
                new_temps[src_idx] += self._heat_source_intensity

        # Step 4: Cooling
        for i in range(CELL_COUNT):
            if cooling[i]:
                new_temps[i] -= self._cooling_effect

        # Step 5: Diffusion (Laplacian)
        # Compute diffusion delta from *original* temperatures to avoid order bias
        diffusion_delta = [0.0] * CELL_COUNT
        for i in range(CELL_COUNT):
            nbrs = self._neighbor_cache[i]
            if nbrs:
                avg_neighbor = sum(state.temperatures[n] for n in nbrs) / len(nbrs)
                diffusion_delta[i] = self._diffusion_rate * (avg_neighbor - state.temperatures[i])

        for i in range(CELL_COUNT):
            new_temps[i] += diffusion_delta[i]

        # Step 6: Clamp
        for i in range(CELL_COUNT):
            new_temps[i] = max(0.0, min(1.0, round(new_temps[i], 6)))

        return ThermalGridState(
            temperatures=new_temps,
            cooling_active=cooling,
            global_cooling=global_cooling,
        )

    def factual_transition(
        self,
        *,
        intervention: str,
        external_input: float,
    ) -> ScenarioTransition:
        self._state = self._compute_transition(
            self._state,
            intervention=intervention,
            external_input=external_input,
        )
        agg = _compute_aggregates(self._state.temperatures, self._alarm_threshold)
        level = self._compute_level(agg["hotspot_fraction"])

        return ScenarioTransition(
            state={
                "mean_temp": agg["mean_temp"],
                "max_temp": agg["max_temp"],
                "min_temp": agg["min_temp"],
                "hotspot_count": agg["hotspot_count"],
                "hotspot_fraction": agg["hotspot_fraction"],
                "global_cooling": self._state.global_cooling,
                "world_shape": [GRID_ROWS, GRID_COLS],
                "cell_count": CELL_COUNT,
                "grid_temperatures": list(self._state.temperatures),
                "grid_cooling": list(self._state.cooling_active),
            },
            propositions=self.observe().propositions,
            alarm=level == 3,
            level=level,
        )

    def simulate_counterfactual(
        self,
        *,
        intervention: str,
        external_input: float,
    ) -> ScenarioTransition:
        simulated = self._compute_transition(
            self._state,
            intervention=intervention,
            external_input=external_input,
        )
        agg = _compute_aggregates(simulated.temperatures, self._alarm_threshold)
        level = self._compute_level(agg["hotspot_fraction"])

        if level == 3:
            proposition = "GRID_CRITICAL"
        elif level == 2:
            proposition = "GRID_WARNING"
        else:
            proposition = "GRID_NORMAL"

        return ScenarioTransition(
            state={
                "mean_temp": agg["mean_temp"],
                "max_temp": agg["max_temp"],
                "min_temp": agg["min_temp"],
                "hotspot_count": agg["hotspot_count"],
                "hotspot_fraction": agg["hotspot_fraction"],
                "global_cooling": simulated.global_cooling,
                "world_shape": [GRID_ROWS, GRID_COLS],
                "cell_count": CELL_COUNT,
                "grid_temperatures": list(simulated.temperatures),
                "grid_cooling": list(simulated.cooling_active),
            },
            propositions=[proposition],
            alarm=level == 3,
            level=level,
        )

    # ─── Formula / intervention / propositions ───────────────────────────

    def get_formula(self, observation: ScenarioObservation) -> str:
        if observation.level == 3:
            return "GRID_CRITICAL -> ACTIVATE_COOLING"
        if observation.level == 2:
            return "GRID_WARNING -> ACTIVATE_COOLING"
        return "GRID_NORMAL -> KEEP_IDLE"

    def select_intervention(self, observation: ScenarioObservation) -> str:
        if observation.level >= 2:
            return "activate_cooling"
        return "deactivate_cooling"

    def get_main_proposition(self, observation: ScenarioObservation) -> str:
        if observation.level == 3:
            return "GRID_CRITICAL"
        if observation.level == 2:
            return "GRID_WARNING"
        return "GRID_NORMAL"

    def get_intervention_proposition(self, intervention: str) -> str:
        if intervention == "activate_cooling":
            return "ACTIVATE_COOLING"
        return "KEEP_IDLE"

    def evaluate_relation_kind(
        self,
        *,
        factual: ScenarioTransition,
        counterfactual: ScenarioTransition,
    ) -> str:
        """En grilla térmica, menor mean_temp es mejor (como escalar térmico)."""
        factual_val = factual.state.get("mean_temp", 0.0)
        counterfactual_val = counterfactual.state.get("mean_temp", 0.0)
        if factual_val <= counterfactual_val:
            return "support"
        return "contradiction"

    # ─── Serialization helpers ───────────────────────────────────────────

    def to_observation_dict(self, observation: ScenarioObservation) -> Dict[str, Any]:
        """Override para incluir metadata espacial en observación persistida."""
        base = {
            **observation.state,
            "alarm": observation.alarm,
            "world_level": observation.level,
            "propositions": observation.propositions,
            "scenario": self.config.name,
        }
        # Ensure spatial metadata is present
        base.setdefault("world_shape", [GRID_ROWS, GRID_COLS])
        base.setdefault("cell_count", CELL_COUNT)
        return base

    def to_transition_dict(self, transition: ScenarioTransition) -> Dict[str, Any]:
        """Override para incluir metadata espacial en transición persistida."""
        base = {
            **transition.state,
            "alarm": transition.alarm,
            "world_level": transition.level,
        }
        base.setdefault("world_shape", [GRID_ROWS, GRID_COLS])
        base.setdefault("cell_count", CELL_COUNT)
        return base


# ─── Factory ─────────────────────────────────────────────────────────────────

def create_thermal_grid_5x5(
    *,
    initial_temperatures: Optional[List[float]] = None,
    alarm_threshold: float = 0.85,
) -> ThermalGrid5x5Scenario:
    """Crea grilla térmica 5x5 con configuración por defecto."""
    return ThermalGrid5x5Scenario(
        initial_temperatures=initial_temperatures,
        alarm_threshold=alarm_threshold,
    )

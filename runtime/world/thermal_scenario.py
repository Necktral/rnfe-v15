"""Escenario térmico homeostático - implementación de referencia."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

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


@dataclass
class ThermalWorldState:
    """Estado interno del mundo térmico."""

    temperature: float
    cooling_active: bool
    alarm: bool


class ThermalScenario(CognitiveScenario):
    """Escenario homeostático de control de temperatura.

    Este escenario modela un sistema de control de temperatura con tres niveles:
    - Nivel 1 (NORMAL):   temperature < warning_threshold
    - Nivel 2 (WARNING):  warning_threshold ≤ temperature < alarm_threshold
    - Nivel 3 (CRITICAL): temperature ≥ alarm_threshold (alarm=True)

    Los parámetros están normalizados en [0.0, 1.0] como los rangos máximos
    de los parámetros de las familias de razonamientos.
    """

    def __init__(
        self,
        *,
        initial_temperature: float = 0.82,
        alarm_threshold: float = 0.85,
        cooling_effect: float = 0.07,
        warning_threshold: float = 0.60,
    ):
        """Inicializa escenario térmico.

        Args:
            initial_temperature: Temperatura inicial (0.0-1.0).
            alarm_threshold: Umbral de alarma / nivel crítico (nivel 2→3).
            cooling_effect: Efecto del enfriamiento por paso.
            warning_threshold: Umbral de advertencia (nivel 1→2). Default 0.60.
        """
        self._alarm_threshold = alarm_threshold
        self._warning_threshold = warning_threshold
        self._cooling_effect = cooling_effect
        self._state = ThermalWorldState(
            temperature=initial_temperature,
            cooling_active=False,
            alarm=initial_temperature >= alarm_threshold,
        )
        self._config = ScenarioConfig(
            name="thermal_homeostasis",
            description="Control de temperatura homeostático con tres niveles (normal/advertencia/crítico)",
            main_variable="temperature",
            alarm_threshold=alarm_threshold,
            warning_threshold=warning_threshold,
            interventions=["activate_cooling", "deactivate_cooling"],
            formula_template="TEMP_HIGH -> ACTIVATE_COOLING",
            type_context={
                "TEMP_NORMAL": "bool",
                "TEMP_WARNING": "bool",
                "TEMP_HIGH": "bool",
                "ACTIVATE_COOLING": "bool",
                "KEEP_IDLE": "bool",
            },
        )

    @property
    def config(self) -> ScenarioConfig:
        return self._config

    @property
    def structural_profile(self) -> ScenarioStructuralProfile:
        """Perfil estructural para evaluación de compatibilidad."""
        cfg = self._config
        config_blob = json.dumps(
            {
                "name": cfg.name,
                "main_variable": cfg.main_variable,
                "alarm_threshold": cfg.alarm_threshold,
                "interventions": cfg.interventions,
                "formula_template": cfg.formula_template,
                "type_context": cfg.type_context,
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
        """Firma causal completa para morfismos dirigidos."""
        cfg = self._config
        return ScenarioCausalSignature(
            scenario_name=cfg.name,
            scenario_version="1.0",
            observable_variables=frozenset({"temperature", "cooling_active"}),
            control_variables=frozenset({"cooling_active"}),
            main_variable="temperature",
            optimization_direction="minimize",
            causal_polarity="lower_is_better",
            alarm_semantics="threshold_above",
            intervention_effects=(
                InterventionEffect(
                    intervention_name="activate_cooling",
                    target_variable="temperature",
                    expected_direction="-",
                    expected_magnitude=self._cooling_effect,
                    semantic_role="corrective",
                ),
                InterventionEffect(
                    intervention_name="deactivate_cooling",
                    target_variable="temperature",
                    expected_direction="+",
                    expected_magnitude=0.0,
                    semantic_role="neutral",
                ),
            ),
            counterfactual_policy="opposite_intervention",
            counterfactual_variable="temperature",
            causal_edges=(
                CausalEdge(source="external_heat", target="temperature", polarity="+"),
                CausalEdge(source="cooling_active", target="temperature", polarity="-"),
                CausalEdge(source="temperature", target="alarm", polarity="+"),
            ),
            proposition_vocabulary=frozenset({
                "TEMP_HIGH", "TEMP_WARNING", "TEMP_NORMAL",
                "COOLING_ACTIVE", "ACTIVATE_COOLING", "KEEP_IDLE",
            }),
        )

    @property
    def alarm_threshold(self) -> float:
        """Umbral de alarma para compatibilidad con código existente."""
        return self._alarm_threshold

    @property
    def warning_threshold(self) -> float:
        """Umbral de advertencia (nivel 1→2)."""
        return self._warning_threshold

    def _compute_level(self, temperature: float) -> int:
        """Determina el nivel del mundo (1/2/3) según la temperatura.

        Returns:
            1 (NORMAL), 2 (WARNING) o 3 (CRITICAL).

        Note:
            ``warning_threshold=0.0`` deshabilita el nivel 2 — el guard
            ``self._warning_threshold > 0.0`` previene activación cuando
            el umbral no fue configurado explícitamente.
        """
        if temperature >= self._alarm_threshold:
            return 3
        if self._warning_threshold > 0.0 and temperature >= self._warning_threshold:
            return 2
        return 1

    def observe(self) -> ScenarioObservation:
        level = self._compute_level(self._state.temperature)
        if level == 3:
            proposition = "TEMP_HIGH"
        elif level == 2:
            proposition = "TEMP_WARNING"
        else:
            proposition = "TEMP_NORMAL"
        propositions = [proposition]
        if self._state.cooling_active:
            propositions.append("COOLING_ACTIVE")

        return ScenarioObservation(
            state={
                "temperature": self._state.temperature,
                "cooling_active": self._state.cooling_active,
            },
            propositions=propositions,
            alarm=self._state.alarm,
            level=level,
        )

    def _compute_transition(
        self,
        state: ThermalWorldState,
        *,
        intervention: str,
        external_input: float,
    ) -> ThermalWorldState:
        """Computa transición de estado."""
        cooling_active = state.cooling_active
        if intervention == "activate_cooling":
            cooling_active = True
        elif intervention == "deactivate_cooling":
            cooling_active = False

        cooling_delta = self._cooling_effect if cooling_active else 0.0
        next_temp = max(0.0, min(1.0, state.temperature + external_input - cooling_delta))

        return ThermalWorldState(
            temperature=next_temp,
            cooling_active=cooling_active,
            alarm=next_temp >= self._alarm_threshold,
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
        level = self._compute_level(self._state.temperature)
        return ScenarioTransition(
            state={
                "temperature": self._state.temperature,
                "cooling_active": self._state.cooling_active,
            },
            propositions=self.observe().propositions,
            alarm=self._state.alarm,
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
        level = self._compute_level(simulated.temperature)
        if level == 3:
            proposition = "TEMP_HIGH"
        elif level == 2:
            proposition = "TEMP_WARNING"
        else:
            proposition = "TEMP_NORMAL"
        return ScenarioTransition(
            state={
                "temperature": simulated.temperature,
                "cooling_active": simulated.cooling_active,
            },
            propositions=[proposition],
            alarm=simulated.alarm,
            level=level,
        )

    def get_formula(self, observation: ScenarioObservation) -> str:
        if observation.level == 3:
            return "TEMP_HIGH -> ACTIVATE_COOLING"
        if observation.level == 2:
            return "TEMP_WARNING -> ACTIVATE_COOLING"
        return "TEMP_NORMAL -> KEEP_IDLE"

    def select_intervention(self, observation: ScenarioObservation) -> str:
        if observation.level >= 2:
            return "activate_cooling"
        return "deactivate_cooling"

    def get_main_proposition(self, observation: ScenarioObservation) -> str:
        if observation.level == 3:
            return "TEMP_HIGH"
        if observation.level == 2:
            return "TEMP_WARNING"
        return "TEMP_NORMAL"

    def get_intervention_proposition(self, intervention: str) -> str:
        if intervention == "activate_cooling":
            return "ACTIVATE_COOLING"
        return "KEEP_IDLE"


# Factory function para compatibilidad
def create_thermal_scenario(
    *,
    initial_temperature: float = 0.82,
    alarm_threshold: float = 0.85,
) -> ThermalScenario:
    """Crea escenario térmico con configuración por defecto."""
    return ThermalScenario(
        initial_temperature=initial_temperature,
        alarm_threshold=alarm_threshold,
    )

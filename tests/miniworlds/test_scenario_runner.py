"""Tests para escenarios cognitivos parametrizables."""

from pathlib import Path

import pytest

from runtime.storage import StorageConfig, StorageFactory
from runtime.world import (
    CognitiveScenario,
    ThermalScenario,
    ResourceScenario,
    ScenarioEpisodeRunner,
    get_scenario,
    list_scenarios,
    SCENARIO_REGISTRY,
    DEFAULT_SCENARIO,
)


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "scenario.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestScenarioRegistry:
    """Tests para el registro de escenarios."""

    def test_thermal_scenario_is_registered(self):
        """Escenario térmico está registrado."""
        assert "thermal_homeostasis" in SCENARIO_REGISTRY

    def test_resource_scenario_is_registered(self):
        """Escenario de recursos está registrado."""
        assert "resource_management" in SCENARIO_REGISTRY

    def test_default_scenario_is_thermal(self):
        """El escenario por defecto es thermal_homeostasis."""
        assert DEFAULT_SCENARIO == "thermal_homeostasis"

    def test_get_scenario_returns_instance(self):
        """get_scenario retorna instancia del escenario."""
        scenario = get_scenario("thermal_homeostasis")
        assert isinstance(scenario, CognitiveScenario)
        assert isinstance(scenario, ThermalScenario)

    def test_get_scenario_with_kwargs(self):
        """get_scenario acepta kwargs de configuración."""
        scenario = get_scenario("thermal_homeostasis", initial_temperature=0.9)
        obs = scenario.observe()
        assert obs.state["temperature"] == 0.9

    def test_get_scenario_raises_for_unknown(self):
        """get_scenario lanza error para escenario desconocido."""
        with pytest.raises(ValueError, match="no encontrado"):
            get_scenario("unknown_scenario")

    def test_list_scenarios_returns_configs(self):
        """list_scenarios retorna configuraciones."""
        configs = list_scenarios()
        assert "thermal_homeostasis" in configs
        assert "resource_management" in configs
        assert configs["thermal_homeostasis"].main_variable == "temperature"
        assert configs["resource_management"].main_variable == "stock_level"


class TestThermalScenario:
    """Tests para el escenario térmico."""

    def test_observe_returns_observation(self):
        """observe() retorna ScenarioObservation."""
        scenario = ThermalScenario()
        obs = scenario.observe()
        assert "temperature" in obs.state
        assert isinstance(obs.propositions, list)

    def test_factual_transition_updates_state(self):
        """factual_transition() actualiza estado."""
        scenario = ThermalScenario(initial_temperature=0.9)
        obs_before = scenario.observe()
        result = scenario.factual_transition(intervention="activate_cooling", external_input=0.03)
        obs_after = scenario.observe()

        assert result.state["cooling_active"] is True
        assert obs_after.state["temperature"] < obs_before.state["temperature"]

    def test_counterfactual_does_not_mutate_state(self):
        """simulate_counterfactual() no muta estado."""
        scenario = ThermalScenario(initial_temperature=0.9)
        obs_before = scenario.observe()
        _ = scenario.simulate_counterfactual(intervention="activate_cooling", external_input=0.03)
        obs_after = scenario.observe()

        assert obs_before.state["temperature"] == obs_after.state["temperature"]

    def test_get_formula_returns_level_appropriate_formula(self):
        """get_formula() retorna fórmula LOTF según nivel del mundo."""
        # Nivel 1 (NORMAL): temperature < 0.60
        scenario_l1 = ThermalScenario(initial_temperature=0.50)
        obs_l1 = scenario_l1.observe()
        assert obs_l1.level == 1
        assert scenario_l1.get_formula(obs_l1) == "TEMP_NORMAL -> KEEP_IDLE"

        # Nivel 2 (WARNING): 0.60 <= temperature < 0.85
        scenario_l2 = ThermalScenario(initial_temperature=0.70)
        obs_l2 = scenario_l2.observe()
        assert obs_l2.level == 2
        assert "TEMP_WARNING" in scenario_l2.get_formula(obs_l2)

        # Nivel 3 (CRITICAL): temperature >= 0.85
        scenario_l3 = ThermalScenario(initial_temperature=0.90)
        obs_l3 = scenario_l3.observe()
        assert obs_l3.level == 3
        assert "TEMP_HIGH" in scenario_l3.get_formula(obs_l3)

    def test_evaluate_relation_kind_support(self):
        """evaluate_relation_kind() retorna support cuando factual es mejor."""
        scenario = ThermalScenario(initial_temperature=0.9)
        factual = scenario.factual_transition(intervention="activate_cooling", external_input=0.03)

        scenario2 = ThermalScenario(initial_temperature=0.9)
        counterfactual = scenario2.simulate_counterfactual(
            intervention="deactivate_cooling", external_input=0.03
        )

        kind = scenario.evaluate_relation_kind(factual=factual, counterfactual=counterfactual)
        assert kind == "support"


class TestResourceScenario:
    """Tests para el escenario de recursos."""

    def test_observe_returns_observation(self):
        """observe() retorna ScenarioObservation."""
        scenario = ResourceScenario()
        obs = scenario.observe()
        assert "stock_level" in obs.state
        assert isinstance(obs.propositions, list)

    def test_factual_transition_updates_state(self):
        """factual_transition() actualiza estado."""
        scenario = ResourceScenario(initial_stock=0.15)
        obs_before = scenario.observe()
        result = scenario.factual_transition(intervention="start_production", external_input=0.03)
        obs_after = scenario.observe()

        assert result.state["production_active"] is True
        assert obs_after.state["stock_level"] > obs_before.state["stock_level"]

    def test_counterfactual_does_not_mutate_state(self):
        """simulate_counterfactual() no muta estado."""
        scenario = ResourceScenario(initial_stock=0.15)
        obs_before = scenario.observe()
        _ = scenario.simulate_counterfactual(intervention="start_production", external_input=0.03)
        obs_after = scenario.observe()

        assert obs_before.state["stock_level"] == obs_after.state["stock_level"]

    def test_inverse_causality_to_thermal(self):
        """Recursos tiene causalidad inversa al térmico (LOW -> ACTIVATE)."""
        scenario = ResourceScenario(initial_stock=0.15, scarcity_threshold=0.20)
        obs = scenario.observe()

        # En escasez, debe activar producción
        assert obs.alarm is True
        intervention = scenario.select_intervention(obs)
        assert intervention == "start_production"

    def test_evaluate_relation_kind_support_for_resources(self):
        """evaluate_relation_kind() en recursos: más stock es support."""
        scenario = ResourceScenario(initial_stock=0.15)
        factual = scenario.factual_transition(intervention="start_production", external_input=0.03)

        scenario2 = ResourceScenario(initial_stock=0.15)
        counterfactual = scenario2.simulate_counterfactual(
            intervention="stop_production", external_input=0.03
        )

        kind = scenario.evaluate_relation_kind(factual=factual, counterfactual=counterfactual)
        assert kind == "support"


class TestScenarioEpisodeRunner:
    """Tests para el runner de episodios con escenarios."""

    def test_runner_with_default_thermal_scenario(self, tmp_path: Path):
        """Runner con escenario térmico por defecto usa nivel 2 (advertencia) por initial_temperature=0.82."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(storage=storage, run_id="run-thermal-default")
        result = runner.run_episode(external_input=0.05)

        assert result["episode"]["scenario"] == "thermal_homeostasis"
        # initial_temperature=0.82 cae en nivel 2 (WARNING: 0.60-0.85)
        # → Estrato I + Estrato II: ABD, ANA, CAU, CTF, DED, PROB, OPT, PLAN
        assert result["episode"]["world_level"] == 2
        assert result["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB", "OPT", "PLAN"
        ]
        storage.close()

    def test_runner_with_resource_scenario(self, tmp_path: Path):
        """Runner funciona con escenario de recursos."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-resource",
            scenario="resource_management",
        )
        result = runner.run_episode(external_input=0.03)

        assert result["episode"]["scenario"] == "resource_management"
        assert "stock_level" in result["episode"]["context"]["observation"]
        storage.close()

    def test_runner_with_scenario_instance(self, tmp_path: Path):
        """Runner funciona con instancia de escenario."""
        storage = _storage(tmp_path)
        scenario = ThermalScenario(initial_temperature=0.95)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-instance",
            scenario=scenario,
        )
        result = runner.run_episode(external_input=0.02)

        assert result["episode"]["scenario"] == "thermal_homeostasis"
        storage.close()

    def test_runner_persists_events_and_artifacts(self, tmp_path: Path):
        """Runner persiste eventos y artifacts."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-persist-test",
            scenario="thermal_homeostasis",
        )
        result = runner.run_episode(external_input=0.05)

        # Verificar evento
        events = storage.list_events(run_id="run-persist-test", limit=50)
        closed_events = [e for e in events if e.event_type == "episode.closed"]
        assert len(closed_events) >= 1

        # Verificar artifact
        artifact_path = Path(result["artifact"]["abs_path"])
        assert artifact_path.exists()

        storage.close()

    def test_runner_both_scenarios_produce_valid_episodes(self, tmp_path: Path):
        """Ambos escenarios producen episodios válidos."""
        storage = _storage(tmp_path)

        # Térnico
        runner1 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-compare-thermal",
            scenario="thermal_homeostasis",
        )
        result1 = runner1.run_episode(external_input=0.05)

        # Recursos
        runner2 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-compare-resource",
            scenario="resource_management",
        )
        result2 = runner2.run_episode(external_input=0.03)

        # Ambos deben tener episodio cerrado con veredicto válido
        valid_verdicts = ["PASSED", "CONDITIONALLY_PASSED", "certified"]
        assert result1["certification"]["verdict"] in valid_verdicts
        assert result2["certification"]["verdict"] in valid_verdicts

        # Ambos deben tener secuencia de razonamiento
        assert len(result1["episode"]["result"]["reasoning_sequence"]) >= 6
        assert len(result2["episode"]["result"]["reasoning_sequence"]) >= 6

        storage.close()


class TestThreeLevelWorld:
    """Tests del mundo de tres niveles según estratos de razonamientos."""

    def test_thermal_level_1_normal(self):
        """Temperatura < 0.60 → nivel 1 (NORMAL), proposición TEMP_NORMAL."""
        scenario = ThermalScenario(initial_temperature=0.40)
        obs = scenario.observe()
        assert obs.level == 1
        assert "TEMP_NORMAL" in obs.propositions
        assert obs.alarm is False

    def test_thermal_level_2_warning(self):
        """Temperatura en [0.60, 0.85) → nivel 2 (WARNING), proposición TEMP_WARNING."""
        scenario = ThermalScenario(initial_temperature=0.72)
        obs = scenario.observe()
        assert obs.level == 2
        assert "TEMP_WARNING" in obs.propositions
        assert obs.alarm is False

    def test_thermal_level_3_critical(self):
        """Temperatura ≥ 0.85 → nivel 3 (CRITICAL), proposición TEMP_HIGH, alarm=True."""
        scenario = ThermalScenario(initial_temperature=0.90)
        obs = scenario.observe()
        assert obs.level == 3
        assert "TEMP_HIGH" in obs.propositions
        assert obs.alarm is True

    def test_thermal_level_2_activates_cooling_preemptively(self):
        """Nivel 2 (WARNING) activa enfriamiento preventivo."""
        scenario = ThermalScenario(initial_temperature=0.70)
        obs = scenario.observe()
        assert obs.level == 2
        intervention = scenario.select_intervention(obs)
        assert intervention == "activate_cooling"

    def test_thermal_level_1_keeps_idle(self):
        """Nivel 1 (NORMAL) mantiene sistema inactivo."""
        scenario = ThermalScenario(initial_temperature=0.40)
        obs = scenario.observe()
        assert obs.level == 1
        intervention = scenario.select_intervention(obs)
        assert intervention == "deactivate_cooling"

    def test_resource_level_1_adequate(self):
        """Stock > 0.40 → nivel 1 (ADEQUATE), proposición STOCK_ADEQUATE."""
        scenario = ResourceScenario(initial_stock=0.60)
        obs = scenario.observe()
        assert obs.level == 1
        assert "STOCK_ADEQUATE" in obs.propositions
        assert obs.alarm is False

    def test_resource_level_2_low(self):
        """Stock en (0.20, 0.40] → nivel 2 (LOW), proposición STOCK_LOW."""
        scenario = ResourceScenario(initial_stock=0.30)
        obs = scenario.observe()
        assert obs.level == 2
        assert "STOCK_LOW" in obs.propositions
        assert obs.alarm is False

    def test_resource_level_3_critical(self):
        """Stock ≤ 0.20 → nivel 3 (CRITICAL), proposición STOCK_CRITICAL, alarm=True."""
        scenario = ResourceScenario(initial_stock=0.10)
        obs = scenario.observe()
        assert obs.level == 3
        assert "STOCK_CRITICAL" in obs.propositions
        assert obs.alarm is True

    def test_level_aware_runner_uses_stratum_i_at_level1(self, tmp_path: Path):
        """Nivel 1 usa solo Estrato I: 6 familias primarias."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-level1",
            scenario="thermal_homeostasis",
        )
        runner.scenario = ThermalScenario(initial_temperature=0.40)
        result = runner.run_episode(external_input=0.01)
        seq = result["episode"]["result"]["reasoning_sequence"]
        assert seq == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
        assert result["episode"]["world_level"] == 1
        storage.close()

    def test_level_aware_runner_uses_stratum_i_ii_at_level2(self, tmp_path: Path):
        """Nivel 2 usa Estrato I + II: 8 familias (+ OPT, PLAN)."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-level2",
            scenario="thermal_homeostasis",
        )
        result = runner.run_episode(external_input=0.05)
        seq = result["episode"]["result"]["reasoning_sequence"]
        assert seq == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB", "OPT", "PLAN"]
        assert result["episode"]["world_level"] == 2
        storage.close()

    def test_level_aware_runner_uses_all_strata_at_level3(self, tmp_path: Path):
        """Nivel 3 usa todos los estratos: 11 familias (+ DIA_ADV, FAL_GUARD, HEUR)."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-level3",
            scenario="thermal_homeostasis",
        )
        runner.scenario = ThermalScenario(initial_temperature=0.90)
        result = runner.run_episode(external_input=0.01)
        seq = result["episode"]["result"]["reasoning_sequence"]
        assert seq == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB",
            "OPT", "PLAN", "DIA_ADV", "FAL_GUARD", "HEUR",
        ]
        assert result["episode"]["world_level"] == 3
        storage.close()

    def test_transition_updates_level(self):
        """factual_transition retorna nivel actualizado."""
        scenario = ThermalScenario(initial_temperature=0.72)
        obs = scenario.observe()
        assert obs.level == 2
        transition = scenario.factual_transition(
            intervention="activate_cooling", external_input=0.01
        )
        assert transition.level in {1, 2}

    def test_counterfactual_level_is_consistent(self):
        """simulate_counterfactual retorna nivel coherente con temperatura simulada."""
        scenario = ThermalScenario(initial_temperature=0.62)
        cf = scenario.simulate_counterfactual(
            intervention="deactivate_cooling", external_input=0.05
        )
        assert cf.level in {2, 3}
        assert cf.level == scenario._compute_level(cf.state["temperature"])

"""Tests para escenario térmico espacial 5x5.

Cubre:
- Estado de grilla (shape, cell_count, determinismo)
- Observación (agregados, derivación de nivel, proposiciones)
- Transición (factual muta estado, counterfactual no)
- Niveles del mundo (1/2/3 derivados de hotspot_fraction)
- Integración con ScenarioEpisodeRunner
- Benchmark comparativo 1x1 vs 5x5
"""

import time
from pathlib import Path

import pytest

from runtime.storage import StorageConfig, StorageFactory
from runtime.world import (
    CognitiveScenario,
    ThermalScenario,
    ThermalGrid5x5Scenario,
    ScenarioEpisodeRunner,
    get_scenario,
    list_scenarios,
    SCENARIO_REGISTRY,
)
from runtime.world.thermal_grid_scenario import (
    GRID_ROWS,
    GRID_COLS,
    CELL_COUNT,
    _neighbors,
    _compute_aggregates,
)


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "grid5x5.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


# ─── Grid state tests ────────────────────────────────────────────────────────

class TestGridState:
    """Tests para el estado de la grilla."""

    def test_grid_dimensions(self):
        """Grid es 5x5 = 25 celdas."""
        assert GRID_ROWS == 5
        assert GRID_COLS == 5
        assert CELL_COUNT == 25

    def test_scenario_reports_world_shape(self):
        """Config expone world_shape=(5, 5)."""
        s = ThermalGrid5x5Scenario()
        assert s.config.world_shape == (5, 5)
        assert s.grid_shape == (5, 5)
        assert s.cell_count == 25

    def test_default_temperatures_have_25_elements(self):
        """Inicialización por defecto produce 25 temperaturas."""
        s = ThermalGrid5x5Scenario()
        obs = s.observe()
        assert len(obs.state["grid_temperatures"]) == 25
        assert len(obs.state["grid_cooling"]) == 25

    def test_custom_temperatures_accepted(self):
        """Se acepta lista explícita de 25 temperaturas."""
        temps = [0.5] * 25
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert obs.state["grid_temperatures"] == temps

    def test_invalid_temperature_count_raises(self):
        """Error si se pasan ≠25 temperaturas."""
        with pytest.raises(ValueError, match="25"):
            ThermalGrid5x5Scenario(initial_temperatures=[0.5] * 10)

    def test_deterministic_initialization(self):
        """Dos instancias con mismos params producen mismo estado."""
        s1 = ThermalGrid5x5Scenario()
        s2 = ThermalGrid5x5Scenario()
        assert s1.observe().state["grid_temperatures"] == s2.observe().state["grid_temperatures"]

    def test_temperatures_in_valid_range(self):
        """Todas las temperaturas están en [0.0, 1.0]."""
        s = ThermalGrid5x5Scenario()
        obs = s.observe()
        for t in obs.state["grid_temperatures"]:
            assert 0.0 <= t <= 1.0


# ─── Neighbor computation tests ──────────────────────────────────────────────

class TestNeighbors:
    """Tests para cómputo de vecinos Von-Neumann."""

    def test_corner_has_2_neighbors(self):
        """Esquinas tienen 2 vecinos."""
        assert len(_neighbors(0)) == 2   # top-left
        assert len(_neighbors(4)) == 2   # top-right
        assert len(_neighbors(20)) == 2  # bottom-left
        assert len(_neighbors(24)) == 2  # bottom-right

    def test_edge_has_3_neighbors(self):
        """Bordes (no esquinas) tienen 3 vecinos."""
        assert len(_neighbors(2)) == 3   # top-center
        assert len(_neighbors(10)) == 3  # left-center
        assert len(_neighbors(14)) == 3  # right-center
        assert len(_neighbors(22)) == 3  # bottom-center

    def test_center_has_4_neighbors(self):
        """Centro tiene 4 vecinos."""
        assert len(_neighbors(12)) == 4  # dead center

    def test_neighbor_indices_valid(self):
        """Todos los índices de vecinos están en [0, 24]."""
        for i in range(25):
            for n in _neighbors(i):
                assert 0 <= n < 25


# ─── Aggregates tests ────────────────────────────────────────────────────────

class TestAggregates:
    """Tests para agregados globales."""

    def test_aggregates_all_uniform(self):
        """Grilla uniforme: hotspot_count=25 si todos > threshold."""
        temps = [0.90] * 25
        agg = _compute_aggregates(temps, alarm_threshold=0.85)
        assert agg["mean_temp"] == 0.9
        assert agg["max_temp"] == 0.9
        assert agg["min_temp"] == 0.9
        assert agg["hotspot_count"] == 25
        assert agg["hotspot_fraction"] == 1.0

    def test_aggregates_no_hotspots(self):
        """Grilla fría: hotspot_count=0."""
        temps = [0.50] * 25
        agg = _compute_aggregates(temps, alarm_threshold=0.85)
        assert agg["hotspot_count"] == 0
        assert agg["hotspot_fraction"] == 0.0

    def test_aggregates_mixed(self):
        """Grilla mixta: hotspots parciales."""
        temps = [0.90] * 10 + [0.50] * 15
        agg = _compute_aggregates(temps, alarm_threshold=0.85)
        assert agg["hotspot_count"] == 10
        assert abs(agg["hotspot_fraction"] - 0.4) < 0.001

    def test_observation_contains_aggregates(self):
        """Observación del escenario contiene todos los agregados."""
        s = ThermalGrid5x5Scenario()
        obs = s.observe()
        required_keys = {"mean_temp", "max_temp", "min_temp", "hotspot_count", "hotspot_fraction"}
        assert required_keys.issubset(obs.state.keys())


# ─── Level derivation tests ──────────────────────────────────────────────────

class TestLevelDerivation:
    """Tests para derivación de world_level desde agregados."""

    def test_level_1_all_cold(self):
        """hotspot_fraction=0 → nivel 1 (NORMAL)."""
        temps = [0.50] * 25  # all below 0.85
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert obs.level == 1
        assert "GRID_NORMAL" in obs.propositions
        assert obs.alarm is False

    def test_level_2_some_hotspots(self):
        """20% ≤ hotspot_fraction < 40% → nivel 2 (WARNING).

        5/25 = 0.20 → exactly at warning threshold → level 2.
        """
        temps = [0.90] * 5 + [0.50] * 20  # 5/25 = 0.20 hotspot_fraction
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert obs.level == 2
        assert "GRID_WARNING" in obs.propositions
        assert obs.alarm is False

    def test_level_3_many_hotspots(self):
        """hotspot_fraction ≥ 40% → nivel 3 (CRITICAL).

        10/25 = 0.40 → exactly at critical threshold → level 3.
        """
        temps = [0.90] * 10 + [0.50] * 15  # 10/25 = 0.40 hotspot_fraction
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert obs.level == 3
        assert "GRID_CRITICAL" in obs.propositions
        assert obs.alarm is True

    def test_level_3_all_hot(self):
        """Todos los hotspots → nivel 3."""
        temps = [0.95] * 25
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert obs.level == 3
        assert obs.state["hotspot_fraction"] == 1.0

    def test_level_boundary_below_warning(self):
        """4/25 = 0.16 < 0.20 → nivel 1."""
        temps = [0.90] * 4 + [0.50] * 21
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert obs.level == 1

    def test_level_boundary_between_warning_and_critical(self):
        """9/25 = 0.36 → nivel 2 (between 0.20 and 0.40)."""
        temps = [0.90] * 9 + [0.50] * 16
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert obs.level == 2

    def test_custom_thresholds(self):
        """Umbrales personalizados cambian la derivación."""
        temps = [0.90] * 3 + [0.50] * 22  # 3/25 = 0.12
        s = ThermalGrid5x5Scenario(
            initial_temperatures=temps,
            warning_fraction=0.10,
            critical_fraction=0.20,
        )
        obs = s.observe()
        assert obs.level == 2  # 0.12 >= 0.10 warning


# ─── Transition tests ────────────────────────────────────────────────────────

class TestTransitions:
    """Tests para transiciones factual y contrafactual."""

    def test_factual_mutates_state(self):
        """factual_transition() muta estado interno."""
        temps = [0.80] * 25
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs_before = s.observe()
        s.factual_transition(intervention="activate_cooling", external_input=0.03)
        obs_after = s.observe()
        # Cooling reduces temperature
        assert obs_after.state["mean_temp"] < obs_before.state["mean_temp"]

    def test_counterfactual_does_not_mutate(self):
        """simulate_counterfactual() no muta estado."""
        temps = [0.80] * 25
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs_before = s.observe()
        s.simulate_counterfactual(intervention="activate_cooling", external_input=0.03)
        obs_after = s.observe()
        assert obs_before.state["grid_temperatures"] == obs_after.state["grid_temperatures"]

    def test_cooling_reduces_mean_temp(self):
        """Activar enfriamiento reduce temperatura media."""
        temps = [0.80] * 25
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        result = s.factual_transition(intervention="activate_cooling", external_input=0.0)
        assert result.state["mean_temp"] < 0.80

    def test_no_cooling_with_heat_increases_mean_temp(self):
        """Sin enfriamiento + calor externo aumenta temperatura."""
        temps = [0.50] * 25
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        result = s.factual_transition(intervention="deactivate_cooling", external_input=0.05)
        assert result.state["mean_temp"] > 0.50

    def test_heat_sources_add_extra_heat(self):
        """Fuentes de calor añaden calor extra a celdas específicas."""
        temps = [0.50] * 25
        s = ThermalGrid5x5Scenario(
            initial_temperatures=temps,
            heat_sources=[12],  # only center
            heat_source_intensity=0.10,
        )
        result = s.factual_transition(intervention="deactivate_cooling", external_input=0.0)
        # Center cell (12) should be hotter than edges
        assert result.state["grid_temperatures"][12] > result.state["grid_temperatures"][0]

    def test_diffusion_spreads_heat(self):
        """Difusión propaga calor de celdas calientes a vecinas frías."""
        # One hot cell surrounded by cold cells
        temps = [0.30] * 25
        temps[12] = 0.95  # center is hot
        s = ThermalGrid5x5Scenario(
            initial_temperatures=temps,
            diffusion_rate=0.10,
            heat_sources=[],  # no external sources
        )
        result = s.factual_transition(intervention="deactivate_cooling", external_input=0.0)
        # Center should have lost some heat
        assert result.state["grid_temperatures"][12] < 0.95
        # Neighbors should have gained some heat
        for n in [7, 11, 13, 17]:  # Von-Neumann neighbors of center
            assert result.state["grid_temperatures"][n] > 0.30

    def test_temperatures_stay_clamped(self):
        """Temperaturas se mantienen en [0.0, 1.0] después de transición."""
        temps = [0.99] * 25
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        result = s.factual_transition(intervention="deactivate_cooling", external_input=0.10)
        for t in result.state["grid_temperatures"]:
            assert 0.0 <= t <= 1.0

    def test_evaluate_relation_kind(self):
        """evaluate_relation_kind: cooling produce support."""
        temps = [0.80] * 25
        s1 = ThermalGrid5x5Scenario(initial_temperatures=temps)
        factual = s1.factual_transition(intervention="activate_cooling", external_input=0.03)

        s2 = ThermalGrid5x5Scenario(initial_temperatures=temps)
        counterfactual = s2.simulate_counterfactual(
            intervention="deactivate_cooling", external_input=0.03
        )

        kind = s1.evaluate_relation_kind(factual=factual, counterfactual=counterfactual)
        assert kind == "support"


# ─── Formula and intervention tests ──────────────────────────────────────────

class TestFormulaAndIntervention:
    """Tests para fórmulas LOTF e intervenciones."""

    def test_formula_level_1(self):
        """Nivel 1 → GRID_NORMAL -> KEEP_IDLE."""
        temps = [0.50] * 25
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert s.get_formula(obs) == "GRID_NORMAL -> KEEP_IDLE"

    def test_formula_level_2(self):
        """Nivel 2 → GRID_WARNING -> ACTIVATE_COOLING."""
        temps = [0.90] * 5 + [0.50] * 20
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert s.get_formula(obs) == "GRID_WARNING -> ACTIVATE_COOLING"

    def test_formula_level_3(self):
        """Nivel 3 → GRID_CRITICAL -> ACTIVATE_COOLING."""
        temps = [0.90] * 10 + [0.50] * 15
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert s.get_formula(obs) == "GRID_CRITICAL -> ACTIVATE_COOLING"

    def test_select_intervention_level_1(self):
        """Nivel 1 → deactivate_cooling."""
        temps = [0.50] * 25
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert s.select_intervention(obs) == "deactivate_cooling"

    def test_select_intervention_level_2(self):
        """Nivel 2 → activate_cooling (preventivo)."""
        temps = [0.90] * 5 + [0.50] * 20
        s = ThermalGrid5x5Scenario(initial_temperatures=temps)
        obs = s.observe()
        assert s.select_intervention(obs) == "activate_cooling"


# ─── Registry tests ──────────────────────────────────────────────────────────

class TestGridRegistry:
    """Tests para registro de escenario grid."""

    def test_grid_scenario_registered(self):
        """thermal_grid_5x5 está en el registro."""
        assert "thermal_grid_5x5" in SCENARIO_REGISTRY

    def test_get_scenario_returns_grid(self):
        """get_scenario('thermal_grid_5x5') retorna ThermalGrid5x5Scenario."""
        s = get_scenario("thermal_grid_5x5")
        assert isinstance(s, ThermalGrid5x5Scenario)

    def test_get_scenario_alias(self):
        """Alias 'grid_5x5' funciona."""
        s = get_scenario("grid_5x5")
        assert isinstance(s, ThermalGrid5x5Scenario)

    def test_list_scenarios_includes_grid(self):
        """list_scenarios() incluye thermal_grid_5x5."""
        configs = list_scenarios()
        assert "thermal_grid_5x5" in configs
        assert configs["thermal_grid_5x5"].world_shape == (5, 5)

    def test_scalar_scenarios_have_no_world_shape(self):
        """Escenarios escalares reportan world_shape=None."""
        configs = list_scenarios()
        assert configs["thermal_homeostasis"].world_shape is None
        assert configs["resource_management"].world_shape is None


# ─── Structural profile and causal signature ─────────────────────────────────

class TestProfileAndSignature:
    """Tests para perfil estructural y firma causal."""

    def test_structural_profile_valid(self):
        """Perfil estructural tiene campos correctos."""
        s = ThermalGrid5x5Scenario()
        p = s.structural_profile
        assert p.scenario_name == "thermal_grid_5x5"
        assert p.optimization_direction == "minimize"
        assert p.relation_polarity == "lower_is_better"

    def test_causal_signature_includes_grid_metadata(self):
        """Firma causal incluye metadata de grilla."""
        s = ThermalGrid5x5Scenario()
        sig = s.causal_signature
        assert sig.metadata["world_shape"] == [5, 5]
        assert sig.metadata["cell_count"] == 25
        assert "mean_temp" in sig.observable_variables
        assert "hotspot_fraction" in sig.observable_variables


# ─── ScenarioEpisodeRunner integration ───────────────────────────────────────

class TestGridEpisodeRunner:
    """Tests de integración con ScenarioEpisodeRunner."""

    def test_runner_produces_valid_episode(self, tmp_path: Path):
        """Runner con grid 5x5 produce episodio válido."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-grid-5x5",
            scenario="thermal_grid_5x5",
        )
        result = runner.run_episode(external_input=0.03)

        assert result["episode"]["scenario"] == "thermal_grid_5x5"
        assert "mean_temp" in result["episode"]["context"]["observation"]
        assert "hotspot_fraction" in result["episode"]["context"]["observation"]
        assert result["episode"]["context"]["observation"]["cell_count"] == 25
        storage.close()

    def test_runner_with_instance(self, tmp_path: Path):
        """Runner funciona con instancia directa."""
        storage = _storage(tmp_path)
        temps = [0.50] * 25
        scenario = ThermalGrid5x5Scenario(initial_temperatures=temps)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-grid-instance",
            scenario=scenario,
        )
        result = runner.run_episode(external_input=0.02)

        assert result["episode"]["scenario"] == "thermal_grid_5x5"
        valid_verdicts = ["PASSED", "CONDITIONALLY_PASSED", "certified"]
        assert result["certification"]["verdict"] in valid_verdicts
        storage.close()

    def test_runner_baseline_fixed_works(self, tmp_path: Path):
        """baseline_fixed funciona con grid 5x5."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-grid-baseline",
            scenario="thermal_grid_5x5",
            closure_profile="baseline_fixed",
        )
        result = runner.run_episode(external_input=0.03)

        assert result["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB"
        ]
        storage.close()

    def test_runner_level_aware_escalates(self, tmp_path: Path):
        """level_aware escala familias según world_level derivado del grid.

        Con 10/25 hotspots (fraction=0.40) → level 3 → all 11 families.
        """
        storage = _storage(tmp_path)
        temps = [0.90] * 10 + [0.50] * 15
        scenario = ThermalGrid5x5Scenario(initial_temperatures=temps)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-grid-level3",
            scenario=scenario,
            closure_profile="level_aware",
        )
        result = runner.run_episode(external_input=0.02)

        assert result["episode"]["world_level"] == 3
        assert result["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB",
            "OPT", "PLAN", "DIA_ADV", "FAL_GUARD", "HEUR"
        ]
        storage.close()

    def test_runner_persists_artifact(self, tmp_path: Path):
        """Runner persiste artifact con metadata de grilla."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-grid-artifact",
            scenario="thermal_grid_5x5",
        )
        result = runner.run_episode(external_input=0.03)

        artifact_path = Path(result["artifact"]["abs_path"])
        assert artifact_path.exists()

        # Verify artifact metadata includes grid scenario
        assert result["episode"]["scenario_metadata"]["scenario_name"] == "thermal_grid_5x5"
        storage.close()

    def test_runner_scenario_metadata_includes_world_shape(self, tmp_path: Path):
        """Scenario metadata no incluye world_shape explícitamente en
        _build_scenario_metadata (se usa ScenarioConfig), pero sí en observación."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-grid-meta",
            scenario="thermal_grid_5x5",
        )
        result = runner.run_episode(external_input=0.03)

        obs = result["episode"]["context"]["observation"]
        assert obs["world_shape"] == [5, 5]
        assert obs["cell_count"] == 25
        storage.close()


# ─── Benchmark: 1x1 vs 5x5 ──────────────────────────────────────────────────

class TestBenchmark1x1vs5x5:
    """Benchmark comparativo 1x1 vs 5x5.

    No es un benchmark de rendimiento estricto; es un test que verifica
    que ambos mundos son funcionales y mide la diferencia de costo.
    """

    def test_both_worlds_produce_valid_episodes(self, tmp_path: Path):
        """Ambos mundos (1x1 y 5x5) producen episodios válidos y certificados."""
        storage = _storage(tmp_path)

        # 1x1 thermal
        runner_1x1 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="bench-1x1",
            scenario="thermal_homeostasis",
        )
        result_1x1 = runner_1x1.run_episode(external_input=0.05)

        # 5x5 thermal
        runner_5x5 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="bench-5x5",
            scenario="thermal_grid_5x5",
        )
        result_5x5 = runner_5x5.run_episode(external_input=0.03)

        valid_verdicts = ["PASSED", "CONDITIONALLY_PASSED", "certified"]
        assert result_1x1["certification"]["verdict"] in valid_verdicts
        assert result_5x5["certification"]["verdict"] in valid_verdicts

        # Both have reasoning sequences
        assert len(result_1x1["episode"]["result"]["reasoning_sequence"]) >= 6
        assert len(result_5x5["episode"]["result"]["reasoning_sequence"]) >= 6

        storage.close()

    def test_timing_comparison(self, tmp_path: Path):
        """Mide diferencia de tiempo entre 1x1 y 5x5 (no falla, solo mide)."""
        storage = _storage(tmp_path)
        n_episodes = 5

        # Warm up
        warm = ScenarioEpisodeRunner(
            storage=storage, run_id="bench-warm", scenario="thermal_homeostasis",
        )
        warm.run_episode(external_input=0.04)

        # 1x1 timing
        times_1x1 = []
        for i in range(n_episodes):
            r = ScenarioEpisodeRunner(
                storage=storage, run_id=f"bench-t-1x1-{i}", scenario="thermal_homeostasis",
            )
            t0 = time.perf_counter()
            r.run_episode(external_input=0.04)
            times_1x1.append(time.perf_counter() - t0)

        # 5x5 timing
        times_5x5 = []
        for i in range(n_episodes):
            r = ScenarioEpisodeRunner(
                storage=storage, run_id=f"bench-t-5x5-{i}", scenario="thermal_grid_5x5",
            )
            t0 = time.perf_counter()
            r.run_episode(external_input=0.03)
            times_5x5.append(time.perf_counter() - t0)

        avg_1x1 = sum(times_1x1) / len(times_1x1)
        avg_5x5 = sum(times_5x5) / len(times_5x5)
        ratio = avg_5x5 / avg_1x1 if avg_1x1 > 0 else float("inf")

        # Just print metrics; don't fail
        print(f"\n  1x1 avg: {avg_1x1:.4f}s")
        print(f"  5x5 avg: {avg_5x5:.4f}s")
        print(f"  ratio 5x5/1x1: {ratio:.2f}x")

        # Sanity: 5x5 should not be more than 10x slower
        assert ratio < 10.0, f"5x5 is {ratio:.1f}x slower than 1x1 — too expensive"
        storage.close()

    def test_artifact_size_comparison(self, tmp_path: Path):
        """Compara tamaño de artifact entre 1x1 y 5x5."""
        storage = _storage(tmp_path)

        # 1x1
        r1 = ScenarioEpisodeRunner(
            storage=storage, run_id="bench-size-1x1", scenario="thermal_homeostasis",
        )
        res1 = r1.run_episode(external_input=0.04)
        size_1x1 = Path(res1["artifact"]["abs_path"]).stat().st_size

        # 5x5
        r5 = ScenarioEpisodeRunner(
            storage=storage, run_id="bench-size-5x5", scenario="thermal_grid_5x5",
        )
        res5 = r5.run_episode(external_input=0.03)
        size_5x5 = Path(res5["artifact"]["abs_path"]).stat().st_size

        ratio = size_5x5 / size_1x1 if size_1x1 > 0 else float("inf")
        print(f"\n  1x1 artifact: {size_1x1} bytes")
        print(f"  5x5 artifact: {size_5x5} bytes")
        print(f"  ratio: {ratio:.2f}x")

        # 5x5 artifact should be larger but not absurdly so
        assert size_5x5 > size_1x1, "5x5 artifact should be larger than 1x1"
        assert ratio < 20.0, f"5x5 artifact is {ratio:.1f}x larger — too heavy"
        storage.close()

    def test_trace_length_comparison(self, tmp_path: Path):
        """Compara longitud de traza (reasoning sequence) entre 1x1 y 5x5.

        Con baseline_fixed, ambos deben tener la misma secuencia exacta.
        """
        storage = _storage(tmp_path)

        r1 = ScenarioEpisodeRunner(
            storage=storage, run_id="bench-trace-1x1",
            scenario="thermal_homeostasis", closure_profile="baseline_fixed",
        )
        res1 = r1.run_episode(external_input=0.04)

        r5 = ScenarioEpisodeRunner(
            storage=storage, run_id="bench-trace-5x5",
            scenario="thermal_grid_5x5", closure_profile="baseline_fixed",
        )
        res5 = r5.run_episode(external_input=0.03)

        seq1 = res1["episode"]["result"]["reasoning_sequence"]
        seq5 = res5["episode"]["result"]["reasoning_sequence"]

        # Under baseline_fixed, both must have canonical 6-family sequence
        assert seq1 == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
        assert seq5 == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
        storage.close()

"""Tests de paridad y comparabilidad entre mundo escalar 1x1 y grilla 5x5.

Verifica que:
- Ambos mundos siguen el mismo contrato CognitiveScenario.
- Ambos producen episodios con la misma estructura.
- No hay confusión de identidad entre escalar y espacial.
- baseline_fixed se comporta idénticamente en ambos.
- strict_same_scenario impide contaminación cruzada.
"""

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world import (
    ThermalScenario,
    ThermalGrid5x5Scenario,
    ScenarioEpisodeRunner,
)
from runtime.world.compatibility import ScenarioCompatibilityGraph


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "parity.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestGridVsScalarParity:
    """Paridad contractual entre mundo escalar y grilla."""

    def test_both_implement_cognitive_scenario(self):
        """Ambos implementan CognitiveScenario completa."""
        s1 = ThermalScenario()
        s5 = ThermalGrid5x5Scenario()

        # Both have the full CognitiveScenario API
        for scenario in [s1, s5]:
            assert hasattr(scenario, "config")
            assert hasattr(scenario, "observe")
            assert hasattr(scenario, "factual_transition")
            assert hasattr(scenario, "simulate_counterfactual")
            assert hasattr(scenario, "get_formula")
            assert hasattr(scenario, "select_intervention")
            assert hasattr(scenario, "get_main_proposition")
            assert hasattr(scenario, "get_intervention_proposition")
            assert hasattr(scenario, "evaluate_relation_kind")
            assert hasattr(scenario, "structural_profile")
            assert hasattr(scenario, "causal_signature")

    def test_scenario_names_are_distinct(self):
        """Escenarios tienen nombres distintos (no se confunden)."""
        s1 = ThermalScenario()
        s5 = ThermalGrid5x5Scenario()
        assert s1.config.name != s5.config.name
        assert s1.config.name == "thermal_homeostasis"
        assert s5.config.name == "thermal_grid_5x5"

    def test_world_shape_distinguishes_scalar_from_grid(self):
        """world_shape es None para escalar y (5,5) para grid."""
        s1 = ThermalScenario()
        s5 = ThermalGrid5x5Scenario()
        assert s1.config.world_shape is None
        assert s5.config.world_shape == (5, 5)

    def test_both_produce_valid_observations(self):
        """Ambos producen observaciones con level y propositions."""
        s1 = ThermalScenario()
        s5 = ThermalGrid5x5Scenario()

        for s in [s1, s5]:
            obs = s.observe()
            assert hasattr(obs, "state")
            assert hasattr(obs, "propositions")
            assert hasattr(obs, "alarm")
            assert hasattr(obs, "level")
            assert obs.level in (1, 2, 3)
            assert isinstance(obs.propositions, list)
            assert len(obs.propositions) >= 1

    def test_baseline_fixed_produces_identical_sequence(self, tmp_path: Path):
        """baseline_fixed produce la misma secuencia canónica en ambos."""
        storage = _storage(tmp_path)

        r1 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-baseline-1x1",
            scenario="thermal_homeostasis", closure_profile="baseline_fixed",
        )
        res1 = r1.run_episode(external_input=0.04)

        r5 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-baseline-5x5",
            scenario="thermal_grid_5x5", closure_profile="baseline_fixed",
        )
        res5 = r5.run_episode(external_input=0.03)

        canonical = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
        assert res1["episode"]["result"]["reasoning_sequence"] == canonical
        assert res5["episode"]["result"]["reasoning_sequence"] == canonical
        storage.close()

    def test_both_produce_certification(self, tmp_path: Path):
        """Ambos producen certificación válida."""
        storage = _storage(tmp_path)

        r1 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-cert-1x1",
            scenario="thermal_homeostasis",
        )
        res1 = r1.run_episode(external_input=0.04)

        r5 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-cert-5x5",
            scenario="thermal_grid_5x5",
        )
        res5 = r5.run_episode(external_input=0.03)

        valid = {"certified", "PASSED", "CONDITIONALLY_PASSED"}
        assert res1["certification"]["verdict"] in valid
        assert res5["certification"]["verdict"] in valid
        storage.close()

    def test_both_produce_artifacts(self, tmp_path: Path):
        """Ambos materializan artifacts."""
        storage = _storage(tmp_path)

        r1 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-art-1x1",
            scenario="thermal_homeostasis",
        )
        res1 = r1.run_episode(external_input=0.04)

        r5 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-art-5x5",
            scenario="thermal_grid_5x5",
        )
        res5 = r5.run_episode(external_input=0.03)

        assert Path(res1["artifact"]["abs_path"]).exists()
        assert Path(res5["artifact"]["abs_path"]).exists()
        storage.close()

    def test_episode_payload_structure_matches(self, tmp_path: Path):
        """Ambos tienen la misma estructura top-level de episodio."""
        storage = _storage(tmp_path)

        r1 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-struct-1x1",
            scenario="thermal_homeostasis",
        )
        res1 = r1.run_episode(external_input=0.04)

        r5 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-struct-5x5",
            scenario="thermal_grid_5x5",
        )
        res5 = r5.run_episode(external_input=0.03)

        # Same top-level keys
        ep1 = res1["episode"]
        ep5 = res5["episode"]
        required_keys = {"episode_id", "timestamp", "scenario", "scenario_metadata",
                         "closure_profile", "world_level", "context", "result", "trace"}
        assert required_keys.issubset(ep1.keys())
        assert required_keys.issubset(ep5.keys())
        storage.close()

    def test_compatibility_graph_shows_analogical(self):
        """Grafo de compatibilidad clasifica 1x1 vs 5x5 correctamente.

        Same optimization direction but different names → should be compatible
        or analogical, not equivalent.
        """
        s1 = ThermalScenario()
        s5 = ThermalGrid5x5Scenario()

        graph = ScenarioCompatibilityGraph()
        assessment = graph.assess(s1.structural_profile, s5.structural_profile)

        # Not equivalent (different scenario names)
        assert assessment.compatibility_class != "equivalent"
        # Should be compatible or analogical (same optimization direction, same topology)
        assert assessment.compatibility_class in ("compatible", "analogical")
        # Transfer should be allowed
        assert assessment.transfer_allowed is True

    def test_grid_metadata_not_in_scalar(self, tmp_path: Path):
        """Grid-specific fields (grid_temperatures, cell_count) no están en escalar."""
        storage = _storage(tmp_path)

        r1 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-nomix-1x1",
            scenario="thermal_homeostasis",
        )
        res1 = r1.run_episode(external_input=0.04)

        obs1 = res1["episode"]["context"]["observation"]
        assert "grid_temperatures" not in obs1
        assert "cell_count" not in obs1
        assert "world_shape" not in obs1
        storage.close()

    def test_scalar_metadata_not_confused_in_grid(self, tmp_path: Path):
        """Grid observation uses mean_temp as main_variable, not raw temperature."""
        storage = _storage(tmp_path)

        r5 = ScenarioEpisodeRunner(
            storage=storage, run_id="parity-nomix-5x5",
            scenario="thermal_grid_5x5",
        )
        res5 = r5.run_episode(external_input=0.03)

        obs5 = res5["episode"]["context"]["observation"]
        # Grid uses mean_temp, not temperature
        assert "mean_temp" in obs5
        assert "cell_count" in obs5
        assert obs5["cell_count"] == 25
        storage.close()

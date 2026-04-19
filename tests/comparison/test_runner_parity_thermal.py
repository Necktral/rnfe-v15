"""Tests de paridad entre ScenarioEpisodeRunner y MinimalCognitiveEpisodeRunner.

Verifica que el runner basado en escenarios produce resultados comparables
al runner legacy para el caso térmico de referencia.
"""

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner
from runtime.world.scenario_runner import ScenarioEpisodeRunner


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


class TestRunnerParityThermal:
    """ScenarioEpisodeRunner(thermal) produces parity with legacy runner."""

    def test_both_runners_produce_valid_closure(self, tmp_path: Path):
        """Both runners produce episodes with valid closure."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-legacy")
        res_legacy = legacy.run_episode(external_heat=0.05)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-scenario",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.05)

        # Both should have valid reasoning sequences
        # Default is baseline_fixed → same canonical 6-family sequence
        assert res_legacy["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB",
        ]
        assert res_scenario["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB",
        ]
        storage.close()

    def test_both_runners_produce_comparable_traces(self, tmp_path: Path):
        """Both runners produce traces of equal length."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-trace-l")
        res_legacy = legacy.run_episode(external_heat=0.04)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-trace-s",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.04)

        trace_legacy = res_legacy["episode"]["trace"]
        trace_scenario = res_scenario["episode"]["trace"]

        # Default is baseline_fixed → same trace length as legacy
        assert len(trace_legacy) == len(trace_scenario)
        storage.close()

    def test_both_runners_materialize_artifact(self, tmp_path: Path):
        """Both runners produce materialized artifacts."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-art-l")
        res_legacy = legacy.run_episode(external_heat=0.04)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-art-s",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.04)

        assert Path(res_legacy["artifact"]["abs_path"]).exists()
        assert Path(res_scenario["artifact"]["abs_path"]).exists()
        storage.close()

    def test_both_runners_produce_certification(self, tmp_path: Path):
        """Both runners go through certification without degradation."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-cert-l")
        res_legacy = legacy.run_episode(external_heat=0.04)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-cert-s",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.04)

        valid_verdicts = {"certified", "PASSED", "CONDITIONALLY_PASSED"}
        assert res_legacy["certification"]["verdict"] in valid_verdicts
        assert res_scenario["certification"]["verdict"] in valid_verdicts
        storage.close()

    def test_scenario_runner_adds_metadata_legacy_does_not(self, tmp_path: Path):
        """ScenarioEpisodeRunner adds scenario_metadata that legacy doesn't."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-meta-l")
        res_legacy = legacy.run_episode(external_heat=0.04)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-meta-s",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.04)

        # Legacy doesn't have scenario_metadata
        assert "scenario_metadata" not in res_legacy["episode"]

        # Scenario runner does
        assert "scenario_metadata" in res_scenario["episode"]
        assert res_scenario["episode"]["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
        storage.close()

    def test_baseline_fixed_profile_matches_legacy_closure_contract(self, tmp_path: Path):
        """ScenarioEpisodeRunner with baseline_fixed produces same closure contract as legacy."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-cp-l")
        res_legacy = legacy.run_episode(external_heat=0.04)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-cp-s",
            scenario="thermal_homeostasis",
            closure_profile="baseline_fixed",
        )
        res_scenario = scenario.run_episode(external_input=0.04)

        # Both must produce the canonical sequence
        assert res_legacy["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB",
        ]
        assert res_scenario["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB",
        ]

        # Scenario runner declares the closure profile
        assert res_scenario["episode"]["closure_profile"] == "baseline_fixed"
        storage.close()

    def test_adaptive_min_profile_does_not_break_canonical_sequence(self, tmp_path: Path):
        """adaptive_min profile still passes when sequence is the canonical one."""
        storage = _storage(tmp_path)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-adapt",
            scenario="thermal_homeostasis",
            closure_profile="adaptive_min",
        )
        res = scenario.run_episode(external_input=0.04)

        # canonical sequence should still pass under adaptive_min
        assert res["episode"]["closure_profile"] == "adaptive_min"
        valid_verdicts = {"certified", "PASSED", "CONDITIONALLY_PASSED"}
        assert res["certification"]["verdict"] in valid_verdicts
        storage.close()


class TestLevelAwareExperimental:
    """Tests for level_aware closure profile — experimental, not default.

    These tests validate that level_aware works correctly when explicitly
    requested, but are separate from historical parity to avoid confusion.
    """

    def test_level_aware_escalates_for_warning_temperature(self, tmp_path: Path):
        """level_aware with initial_temperature=0.82 (level 2) produces Stratum I + II."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-la-warning",
            scenario="thermal_homeostasis",
            closure_profile="level_aware",
        )
        result = runner.run_episode(external_input=0.05)

        assert result["episode"]["closure_profile"] == "level_aware"
        assert result["episode"]["world_level"] == 2
        assert result["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB", "OPT", "PLAN",
        ]
        storage.close()

    def test_level_aware_produces_valid_certification(self, tmp_path: Path):
        """level_aware still passes certification."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-la-cert",
            scenario="thermal_homeostasis",
            closure_profile="level_aware",
        )
        result = runner.run_episode(external_input=0.04)

        valid_verdicts = {"certified", "PASSED", "CONDITIONALLY_PASSED"}
        assert result["certification"]["verdict"] in valid_verdicts
        storage.close()

    def test_level_aware_includes_more_families_than_baseline(self, tmp_path: Path):
        """level_aware at level 2 includes more families than baseline_fixed."""
        storage = _storage(tmp_path)

        runner_base = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-la-base",
            scenario="thermal_homeostasis",
            closure_profile="baseline_fixed",
        )
        res_base = runner_base.run_episode(external_input=0.04)

        runner_la = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-la-level",
            scenario="thermal_homeostasis",
            closure_profile="level_aware",
        )
        res_la = runner_la.run_episode(external_input=0.04)

        seq_base = res_base["episode"]["result"]["reasoning_sequence"]
        seq_la = res_la["episode"]["result"]["reasoning_sequence"]

        # baseline_fixed: 6 families, level_aware at level 2: 8 families
        assert len(seq_base) == 6
        assert len(seq_la) == 8
        # level_aware includes all baseline families plus extras
        assert seq_la[:6] == seq_base
        storage.close()

    def test_level_aware_is_not_the_default(self, tmp_path: Path):
        """Verify level_aware must be explicitly requested — default is baseline_fixed."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-la-default-check",
            scenario="thermal_homeostasis",
        )
        assert runner.closure_profile == "baseline_fixed"
        storage.close()

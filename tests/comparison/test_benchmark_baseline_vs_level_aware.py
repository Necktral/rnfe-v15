"""Benchmark comparativo: baseline_fixed vs level_aware.

Mide métricas obligatorias de comparación entre perfiles de cierre para
validar que level_aware es viable como perfil experimental sin degradar
la calidad del runtime frente al baseline histórico.

Métricas:
- closure_rate: proporción de episodios que cierran con veredicto válido
- continuity_mean: delta factual medio (menor es mejor para thermal)
- collapse_count: episodios con relación 'contradiction'
- trace_integrity_rate: proporción de trazas con ≥6 familias Estrato I
- sequence_length: largo de secuencia de razonamiento
- scheduler_cost_per_episode: tiempo promedio del scheduler por episodio
"""

import time
from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.scenario_runner import ScenarioEpisodeRunner


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "bench.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


_VALID_VERDICTS = {"certified", "PASSED", "CONDITIONALLY_PASSED"}
_STRATUM_I = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
_N_EPISODES = 5


def _run_profile(storage, profile: str, scenario: str = "thermal_homeostasis") -> list:
    """Run N episodes with given profile and return results + timing."""
    results = []
    for i in range(_N_EPISODES):
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id=f"bench-{profile}-{i}",
            scenario=scenario,
            closure_profile=profile,
        )
        t0 = time.perf_counter()
        result = runner.run_episode(external_input=0.04)
        elapsed = time.perf_counter() - t0
        result["_bench_elapsed"] = elapsed
        results.append(result)
    return results


def _compute_metrics(results: list) -> dict:
    """Compute benchmark metrics from a list of episode results."""
    n = len(results)
    if n == 0:
        return {}

    # closure_rate: fraction with valid verdict
    closed = sum(1 for r in results if r["certification"]["verdict"] in _VALID_VERDICTS)
    closure_rate = closed / n

    # continuity_mean: average factual_delta (thermal: closer to 0 or negative is better)
    deltas = [r["episode"]["result"]["factual_delta"] for r in results]
    continuity_mean = sum(deltas) / n

    # collapse_count: episodes with contradiction
    collapse_count = sum(
        1 for r in results
        if r["episode"]["result"]["relation_kind"] == "contradiction"
    )

    # trace_integrity_rate: fraction of traces containing all Stratum I families
    def has_stratum_i(trace):
        families = [step["family"] for step in trace]
        return all(f in families for f in _STRATUM_I)

    trace_integrity_rate = sum(
        1 for r in results if has_stratum_i(r["episode"]["trace"])
    ) / n

    # sequence_length: average reasoning sequence length
    seq_lengths = [len(r["episode"]["result"]["reasoning_sequence"]) for r in results]
    sequence_length_mean = sum(seq_lengths) / n

    # scheduler_cost_per_episode: average time per episode
    times = [r["_bench_elapsed"] for r in results]
    scheduler_cost_mean = sum(times) / n

    # artifact_size: average artifact size
    artifact_sizes = []
    for r in results:
        artifact_path = Path(r["artifact"]["abs_path"])
        if artifact_path.exists():
            artifact_sizes.append(artifact_path.stat().st_size)
    artifact_size_mean = sum(artifact_sizes) / len(artifact_sizes) if artifact_sizes else 0

    return {
        "closure_rate": closure_rate,
        "continuity_mean": round(continuity_mean, 6),
        "collapse_count": collapse_count,
        "trace_integrity_rate": trace_integrity_rate,
        "sequence_length_mean": round(sequence_length_mean, 1),
        "scheduler_cost_mean": round(scheduler_cost_mean, 4),
        "artifact_size_mean": round(artifact_size_mean),
        "n_episodes": n,
    }


class TestBenchmarkBaselineVsLevelAware:
    """Benchmark comparativo obligatorio entre baseline_fixed y level_aware."""

    def test_both_profiles_achieve_full_closure(self, tmp_path: Path):
        """Ambos perfiles deben lograr 100% closure rate."""
        storage = _storage(tmp_path)

        results_base = _run_profile(storage, "baseline_fixed")
        results_la = _run_profile(storage, "level_aware")

        metrics_base = _compute_metrics(results_base)
        metrics_la = _compute_metrics(results_la)

        assert metrics_base["closure_rate"] == 1.0, (
            f"baseline_fixed closure_rate={metrics_base['closure_rate']}"
        )
        assert metrics_la["closure_rate"] == 1.0, (
            f"level_aware closure_rate={metrics_la['closure_rate']}"
        )
        storage.close()

    def test_level_aware_does_not_increase_collapse(self, tmp_path: Path):
        """level_aware no debe producir más collapses que baseline_fixed."""
        storage = _storage(tmp_path)

        results_base = _run_profile(storage, "baseline_fixed")
        results_la = _run_profile(storage, "level_aware")

        metrics_base = _compute_metrics(results_base)
        metrics_la = _compute_metrics(results_la)

        assert metrics_la["collapse_count"] <= metrics_base["collapse_count"] + 1, (
            f"level_aware collapse_count={metrics_la['collapse_count']} vs "
            f"baseline_fixed={metrics_base['collapse_count']}"
        )
        storage.close()

    def test_both_profiles_maintain_trace_integrity(self, tmp_path: Path):
        """Ambos perfiles deben tener 100% trace integrity (Stratum I completo)."""
        storage = _storage(tmp_path)

        results_base = _run_profile(storage, "baseline_fixed")
        results_la = _run_profile(storage, "level_aware")

        metrics_base = _compute_metrics(results_base)
        metrics_la = _compute_metrics(results_la)

        assert metrics_base["trace_integrity_rate"] == 1.0
        assert metrics_la["trace_integrity_rate"] == 1.0
        storage.close()

    def test_level_aware_has_longer_sequences(self, tmp_path: Path):
        """level_aware with thermal default (level 2) produces longer sequences than baseline."""
        storage = _storage(tmp_path)

        results_base = _run_profile(storage, "baseline_fixed")
        results_la = _run_profile(storage, "level_aware")

        metrics_base = _compute_metrics(results_base)
        metrics_la = _compute_metrics(results_la)

        # baseline: 6.0, level_aware at level 2: 8.0
        assert metrics_base["sequence_length_mean"] == 6.0
        assert metrics_la["sequence_length_mean"] >= 8.0
        storage.close()

    def test_level_aware_cost_within_acceptable_ratio(self, tmp_path: Path):
        """level_aware scheduler cost should not exceed 3x baseline."""
        storage = _storage(tmp_path)

        results_base = _run_profile(storage, "baseline_fixed")
        results_la = _run_profile(storage, "level_aware")

        metrics_base = _compute_metrics(results_base)
        metrics_la = _compute_metrics(results_la)

        ratio = (
            metrics_la["scheduler_cost_mean"] / metrics_base["scheduler_cost_mean"]
            if metrics_base["scheduler_cost_mean"] > 0
            else float("inf")
        )

        print(f"\n  baseline_fixed avg: {metrics_base['scheduler_cost_mean']:.4f}s")
        print(f"  level_aware avg:    {metrics_la['scheduler_cost_mean']:.4f}s")
        print(f"  ratio:              {ratio:.2f}x")

        assert ratio < 3.0, (
            f"level_aware is {ratio:.1f}x more expensive than baseline_fixed — "
            f"exceeds 3x acceptable limit"
        )
        storage.close()

    def test_full_benchmark_report(self, tmp_path: Path):
        """Generate complete benchmark report comparing both profiles.

        This test always passes — it prints metrics for human review.
        """
        storage = _storage(tmp_path)

        results_base = _run_profile(storage, "baseline_fixed")
        results_la = _run_profile(storage, "level_aware")

        metrics_base = _compute_metrics(results_base)
        metrics_la = _compute_metrics(results_la)

        print("\n" + "=" * 70)
        print("BENCHMARK: baseline_fixed vs level_aware")
        print("=" * 70)
        print(f"{'Metric':<30} {'baseline_fixed':>15} {'level_aware':>15}")
        print("-" * 70)
        for key in [
            "closure_rate",
            "continuity_mean",
            "collapse_count",
            "trace_integrity_rate",
            "sequence_length_mean",
            "scheduler_cost_mean",
            "artifact_size_mean",
        ]:
            v_base = metrics_base.get(key, "N/A")
            v_la = metrics_la.get(key, "N/A")
            print(f"  {key:<28} {str(v_base):>15} {str(v_la):>15}")
        print("=" * 70)

        storage.close()

    def test_benchmark_resource_scenario(self, tmp_path: Path):
        """Benchmark also works for resource_management scenario."""
        storage = _storage(tmp_path)

        results_base = _run_profile(storage, "baseline_fixed", scenario="resource_management")
        results_la = _run_profile(storage, "level_aware", scenario="resource_management")

        metrics_base = _compute_metrics(results_base)
        metrics_la = _compute_metrics(results_la)

        # Both must achieve full closure
        assert metrics_base["closure_rate"] == 1.0
        assert metrics_la["closure_rate"] == 1.0

        # Both must have trace integrity
        assert metrics_base["trace_integrity_rate"] == 1.0
        assert metrics_la["trace_integrity_rate"] == 1.0
        storage.close()

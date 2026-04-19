from __future__ import annotations

import sqlite3
from pathlib import Path
import importlib.util
import sys

import pytest


def _load_runner_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "benchmark_fractal_ablation_5x.py"
    spec = importlib.util.spec_from_file_location("benchmark_fractal_ablation_5x", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sqlite_schema_and_idempotent_insert(tmp_path: Path) -> None:
    mod = _load_runner_module()
    db_path = tmp_path / "ablation.db"

    row = {
        "variant_id": "A1_baseline_puro",
        "label": "A1",
        "bridge_mode": "off",
        "policy_ref": "legacy",
        "bridge_ref": "legacy",
        "metrics_ref": "legacy",
        "FHS": 20.0,
        "SSS": 30.0,
        "OPS": 100.0,
        "common_pass_rate": 1.0,
        "ordered_pass_rate": 1.0,
        "functional_regression": False,
        "seed_metrics_rows": [{"seed": 101, "FHS": 20.0, "SSS": 30.0, "OPS": 100.0}],
        "score_breakdown_rows": [
            {
                "seed": 101,
                "score": "OPS",
                "component": "iters_avg_delta_pct",
                "weight": 0.45,
                "normalized_value": 0.0,
                "contribution": 0.0,
                "clipped": False,
            }
        ],
    }
    report = {
        "timestamp": "20260419-120000",
        "out_dir": str(tmp_path / "out"),
        "stable_ref": "14a3225",
        "legacy_ref": "bde20c9",
        "seeds": [101],
        "bootstrap_samples": 10000,
        "permutation_samples": 50000,
        "workload_scale": 1.0,
        "analysis": {},
    }
    effects = {"A5_vs_A1": {"FHS_delta": {"point": 0.0}, "SSS_delta": {"point": 1.0}, "OPS_delta": {"point": -1.0}}}
    inference = {"quality_flags": {"have_all_variants": False}}
    negative = {"status": "ok", "replicates_effect": False}
    sensitivity = {"status": "ok", "summary": {"stable_positive_fraction": 1.0, "has_large_negative": False}}
    chart_manifest = [
        {
            "chart_id": "scores_by_variant",
            "relative_path": "charts/scores_by_variant.png",
            "sha256": "deadbeef",
            "size_bytes": 123,
            "series": {"x": [1, 2], "y": [3, 4]},
            "created_at": "2026-04-19T12:00:00",
        }
    ]
    family_details = [
        {
            "variant_id": "A1_baseline_puro",
            "seed": 101,
            "family_type": "geometry",
            "family_id": "T",
            "geometry_name": "Sierpinski Triangle",
            "geometry_family": "T",
            "metric": "fd",
            "value": 1.58,
            "applicable": True,
        },
        {
            "variant_id": "A1_baseline_puro",
            "seed": 101,
            "family_type": "reasoning",
            "family_id": "cau",
            "geometry_name": "Sierpinski Triangle",
            "geometry_family": "T",
            "metric": "present",
            "value": None,
            "present": None,
            "position": None,
            "applicable": False,
        },
    ]
    family_summary = [
        {
            "variant_id": "A1_baseline_puro",
            "seed": 101,
            "family_type": "geometry",
            "family_id": "T",
            "metric": "coverage_geometry_count",
            "mean": 1.0,
            "stdev": 0.0,
            "n": 1,
        }
    ]
    family_diag = {"status": "ok"}

    for _ in range(2):
        mod.persist_run_to_sqlite(
            db_path=db_path,
            run_id="run-x",
            report=report,
            rows=[row],
            effects=effects,
            inference_diagnostics=inference,
            negative_controls=negative,
            sensitivity=sensitivity,
            category="No concluyente",
            rationale="test",
            chart_manifest=chart_manifest,
            family_details_rows=family_details,
            family_summary_rows=family_summary,
            family_diagnostics=family_diag,
        )

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM variants").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM seed_metrics").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM score_breakdown").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM geometry_family_metrics").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM reasoning_family_metrics").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chart_manifest").fetchone()[0] == 1


def test_summarize_family_runs_covers_12_families() -> None:
    mod = _load_runner_module()
    rows = []
    for idx, family in enumerate(mod.GEOMETRY_FAMILIES):
        rows.append(
            {
                "geometry_name": f"G{idx}",
                "geometry_family": family,
                "fd": 1.5,
                "r2": 0.9,
                "d_q_0": 1.4,
                "d_q_1": 1.5,
                "d_q_2": 1.6,
                "atlas_scores": {f"F{i}": 0.5 for i in range(1, 10)},
                "atlas_overall": 0.5,
                "scheduler_features": {
                    "uncertainty": 0.5,
                    "contradiction_signal": 0.5,
                    "continuity_recent": 0.5,
                    "edge_pressure": 0.5,
                    "causal_risk": 0.5,
                    "symbolic_regularity": 0.5,
                    "law_fit_signal": 0.5,
                },
                "max_steps": 6,
                "sequence_length": 6,
                "reasoning_applicable": True,
                "reasoning_presence": {fam: True for fam in mod.REASONING_FAMILIES},
                "reasoning_position": {fam: i for i, fam in enumerate(mod.REASONING_FAMILIES)},
            }
        )
    summary = mod.summarize_family_runs(
        [{"seed": 101, "rows": rows, "coverage": {"observed_families": mod.GEOMETRY_FAMILIES, "missing_families": [], "n_observed": 12}}],
        bridge_mode="on",
    )
    diag = summary["family_diagnostics"]
    assert diag["status"] == "ok"
    assert len(diag["missing_geometry_families"]) == 0
    assert len(diag["observed_geometry_families"]) == 12


def test_summarize_family_runs_bridge_off_marks_reasoning_not_applicable() -> None:
    mod = _load_runner_module()
    summary = mod.summarize_family_runs(
        [
            {
                "seed": 202,
                "rows": [
                    {
                        "geometry_name": "Sierpinski Triangle",
                        "geometry_family": "T",
                        "fd": 1.58,
                        "r2": 0.95,
                        "d_q_0": 1.4,
                        "d_q_1": 1.5,
                        "d_q_2": 1.6,
                        "atlas_scores": {f"F{i}": 0.55 for i in range(1, 10)},
                        "atlas_overall": 0.55,
                        "scheduler_features": {
                            "uncertainty": 0.5,
                            "contradiction_signal": 0.4,
                            "continuity_recent": 0.8,
                            "edge_pressure": 0.3,
                            "causal_risk": 0.4,
                            "symbolic_regularity": 0.4,
                            "law_fit_signal": 0.3,
                        },
                        "max_steps": None,
                        "sequence_length": None,
                        "reasoning_applicable": False,
                        "reasoning_presence": {fam: None for fam in mod.REASONING_FAMILIES},
                        "reasoning_position": {fam: None for fam in mod.REASONING_FAMILIES},
                    }
                ],
                "coverage": {"observed_families": ["T"], "missing_families": [], "n_observed": 1},
            }
        ],
        bridge_mode="off",
    )
    details = [r for r in summary["family_details_rows"] if r["family_type"] == "reasoning"]
    assert details
    assert all(not r["applicable"] for r in details)


def test_generate_comparative_charts_produces_manifest(tmp_path: Path) -> None:
    mod = _load_runner_module()
    rows = [
        {
            "variant_id": "A1_baseline_puro",
            "FHS": 20.0,
            "SSS": 30.0,
            "OPS": 100.0,
            "score_breakdown_rows": [
                {"score": "OPS", "component": "iters_avg_delta_pct", "contribution": 0.0},
                {"score": "OPS", "component": "p95_worst_delta_pct", "contribution": 0.0},
            ],
        },
        {
            "variant_id": "A4_metrics_legacy",
            "FHS": 20.0,
            "SSS": 40.0,
            "OPS": 95.0,
            "score_breakdown_rows": [
                {"score": "OPS", "component": "iters_avg_delta_pct", "contribution": -1.0},
                {"score": "OPS", "component": "p95_worst_delta_pct", "contribution": -2.0},
            ],
        },
        {
            "variant_id": "A5_fractal_full",
            "FHS": 20.0,
            "SSS": 90.0,
            "OPS": 85.0,
            "score_breakdown_rows": [
                {"score": "OPS", "component": "iters_avg_delta_pct", "contribution": -2.0},
                {"score": "OPS", "component": "p95_worst_delta_pct", "contribution": -3.0},
            ],
        },
    ]
    effects = {
        "A2_vs_A1": {"FHS_delta": {"point": 0.0}, "SSS_delta": {"point": 0.0}, "OPS_delta": {"point": -10.0}},
        "A5_vs_A4": {"FHS_delta": {"point": 0.0}, "SSS_delta": {"point": 50.0}, "OPS_delta": {"point": 5.0}},
        "A5_vs_A1": {"FHS_delta": {"point": 0.0}, "SSS_delta": {"point": 55.0}, "OPS_delta": {"point": -5.0}},
    }
    family_details = []
    for variant_id in ("A1_baseline_puro", "A4_metrics_legacy", "A5_fractal_full"):
        for fam in mod.GEOMETRY_FAMILIES:
            family_details.append(
                {
                    "variant_id": variant_id,
                    "seed": 101,
                    "family_type": "geometry",
                    "family_id": fam,
                    "geometry_name": f"{fam}-geom",
                    "geometry_family": fam,
                    "metric": "atlas_overall",
                    "value": 0.5,
                    "applicable": True,
                }
            )
        for rfam in mod.REASONING_FAMILIES:
            family_details.append(
                {
                    "variant_id": variant_id,
                    "seed": 101,
                    "family_type": "reasoning",
                    "family_id": rfam,
                    "geometry_name": "T-geom",
                    "geometry_family": "T",
                    "metric": "present",
                    "value": 1.0 if variant_id != "A1_baseline_puro" else 0.0,
                    "applicable": variant_id != "A1_baseline_puro",
                }
            )

    manifest = mod.generate_comparative_charts(
        out_dir=tmp_path,
        rows=rows,
        effects=effects,
        family_details_rows=family_details,
    )
    assert len(manifest) >= 3
    for item in manifest:
        assert "chart_id" in item
        assert "relative_path" in item
        assert (tmp_path / item["relative_path"]).exists()

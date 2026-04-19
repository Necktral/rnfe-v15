#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import statistics
import subprocess
import sys
import textwrap
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.reasoning_stress.ablation_stats import (
    bootstrap_ci_delta_abs,
    bootstrap_ci_delta_pct,
    cliffs_delta,
    hedges_g,
    permutation_test_mean_delta,
    wilson_interval,
)


DEFAULT_STABLE_REF = "14a3225"
DEFAULT_LEGACY_REF = "bde20c9"
DEFAULT_OUT_ROOT = REPO_ROOT / "rnfe_artifacts" / "causal_reports"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "reasoning_stress_ablation.db"
DEFAULT_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_PERMUTATION_SAMPLES = 50000

GEOMETRY_FAMILIES = [
    "T",
    "C",
    "F3D",
    "B",
    "AXC",
    "AXD",
    "MC",
    "RS",
    "AC",
    "GF",
    "W",
    "PF",
]

REASONING_FAMILIES = [
    "abd",
    "ana",
    "cau",
    "ctf",
    "ded",
    "prob",
    "heur",
    "dia_adv",
    "fal_guard",
    "eml_sr",
]

COMMON_REGRESSION = (
    "tests/regression/test_reasoning_meta_pipeline.py "
    "tests/regression/test_meta_scheduler_policy_units.py "
    "tests/regression/test_meta_scheduler_storage_trace.py "
    "tests/regression/test_meta_scheduler_adaptive.py"
)

COMMON_STRESS = (
    "tests/reasoning_stress/test_boundary_sweep.py "
    "tests/reasoning_stress/test_pairwise_interaction.py "
    "tests/reasoning_stress/test_hypercube_sampling.py "
    "tests/reasoning_stress/test_adversarial_thresholds.py "
    "tests/reasoning_stress/test_temporal_hysteresis.py "
    "tests/reasoning_stress/test_family_contribution.py "
    "tests/reasoning_stress/test_atlas_comprehensive.py"
)

ORDERED_SUITE = (
    "tests/reasoning_stress/test_pairwise_interaction.py "
    "tests/reasoning_stress/test_boundary_sweep.py "
    "tests/reasoning_stress/test_adversarial_thresholds.py "
    "tests/reasoning_stress/test_temporal_hysteresis.py "
    "tests/reasoning_stress/test_family_contribution.py"
)

STRUCTURAL_PROBES = (
    "tests/reasoning_stress/test_multiscale_boundary.py "
    "tests/reasoning_stress/test_box_counting.py "
    "tests/reasoning_stress/test_activation_avalanche.py "
    "tests/reasoning_stress/test_temporal_cascade.py "
    "tests/reasoning_stress/test_fractal_atlas.py"
)

FRACTAL_ONLY_SUITE = (
    "tests/reasoning_stress/test_geometry_catalog.py "
    "tests/reasoning_stress/test_experiment2_atlas.py "
    "tests/reasoning_stress/test_fractal_atlas.py "
    "tests/reasoning_stress/test_multiscale_boundary.py "
    "tests/reasoning_stress/test_box_counting.py "
    "tests/reasoning_stress/test_activation_avalanche.py "
    "tests/reasoning_stress/test_temporal_cascade.py"
)


POLICY_FILE = "runtime/reasoning/scheduler_meta/policy.py"
BRIDGE_FILE = "tests/reasoning_stress/test_geometry_catalog.py"
METRICS_FILES = [
    "tests/reasoning_stress/fractal_utils.py",
    "tests/reasoning_stress/fractal_geometries.py",
]
PAIRWISE_FIX_FILE = "tests/reasoning_stress/test_pairwise_interaction.py"

SSS_COMPONENT_WEIGHTS: dict[str, float] = {
    "convergent_ratio": 0.26,
    "non_path_mb": 0.18,
    "non_path_bc": 0.16,
    "self_similar": 0.18,
    "avalanche_diversity": 0.12,
    "temporal_stability": 0.10,
}

SSS_COMPONENT_FAMILIES: dict[str, str] = {
    "convergent_ratio": "multiscale",
    "non_path_mb": "multiscale",
    "non_path_bc": "box_counting",
    "self_similar": "temporal",
    "avalanche_diversity": "avalanche",
    "temporal_stability": "temporal",
}


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    label: str
    policy_ref: str  # stable | legacy
    bridge_ref: str  # stable | legacy
    bridge_mode: str  # on | off
    metrics_ref: str  # stable | legacy


VARIANTS: list[VariantSpec] = [
    VariantSpec(
        variant_id="A1_baseline_puro",
        label="A1 baseline puro (policy legacy + bridge off + metrics legacy)",
        policy_ref="legacy",
        bridge_ref="legacy",
        bridge_mode="off",
        metrics_ref="legacy",
    ),
    VariantSpec(
        variant_id="A2_bridge_on_policy_baseline",
        label="A2 bridge estable sobre policy baseline (metrics legacy)",
        policy_ref="legacy",
        bridge_ref="stable",
        bridge_mode="on",
        metrics_ref="legacy",
    ),
    VariantSpec(
        variant_id="A3_policy_modulada_sin_fractal_full",
        label="A3 policy estable sin bridge (metrics legacy)",
        policy_ref="stable",
        bridge_ref="legacy",
        bridge_mode="off",
        metrics_ref="legacy",
    ),
    VariantSpec(
        variant_id="A4_metrics_legacy",
        label="A4 policy+bridge estables con metrics legacy",
        policy_ref="stable",
        bridge_ref="stable",
        bridge_mode="on",
        metrics_ref="legacy",
    ),
    VariantSpec(
        variant_id="A5_fractal_full",
        label="A5 fractal full (policy+bridge+metrics estables)",
        policy_ref="stable",
        bridge_ref="stable",
        bridge_mode="on",
        metrics_ref="stable",
    ),
]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estudio causal 5x: fractal vs no fractal con ablaciones."
    )
    parser.add_argument(
        "--stable-ref",
        default=DEFAULT_STABLE_REF,
        help="Referencia estable (default: 14a3225).",
    )
    parser.add_argument(
        "--legacy-ref",
        default=DEFAULT_LEGACY_REF,
        help="Referencia legacy (default: bde20c9).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Directorio base de salida de artefactos (default: rnfe_artifacts/causal_reports).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Ruta SQLite para persistencia del estudio (default: data/reasoning_stress_ablation.db).",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Lista de seeds separadas por coma.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(v.variant_id for v in VARIANTS),
        help="IDs de variantes separadas por coma.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Muestras bootstrap para CI95 (default: 10000).",
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=DEFAULT_PERMUTATION_SAMPLES,
        help="Muestras de permutación para pruebas no paramétricas (default: 50000).",
    )
    parser.add_argument(
        "--workload-scale",
        type=float,
        default=1.0,
        help="Escala multiplicativa para tamaño de workloads (default: 1.0).",
    )
    parser.add_argument(
        "--skip-suites",
        action="store_true",
        help="Omitir ejecución de suites pytest.",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Omitir probes funcionales/estructurales.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Omitir benchmarks W1/W2/W3.",
    )
    parser.add_argument(
        "--keep-variants",
        action="store_true",
        help="Conservar roots materializados de variantes.",
    )
    return parser.parse_args()


def run_cmd(
    cmd: list[str] | str,
    *,
    cwd: Path,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
    shell: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env["PYTHONHASHSEED"] = "0"
    if env:
        merged_env.update(env)

    if shell:
        proc = subprocess.run(
            cmd if isinstance(cmd, str) else " ".join(cmd),
            cwd=str(cwd),
            env=merged_env,
            shell=True,
            text=True,
            capture_output=True,
            check=check,
        )
    else:
        assert isinstance(cmd, list)
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=merged_env,
            text=True,
            capture_output=True,
            check=check,
        )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            (
                f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n\n"
                f"[stdout]\n{proc.stdout}\n\n"
                f"[stderr]\n{proc.stderr}\n"
            ),
            encoding="utf-8",
        )
    return proc


def parse_seeds(seeds_raw: str) -> list[int]:
    out: list[int] = []
    for chunk in seeds_raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    if not out:
        raise ValueError("La lista de seeds no puede estar vacía.")
    return out


def resolve_variants(variant_ids_raw: str) -> list[VariantSpec]:
    wanted = {v.strip() for v in variant_ids_raw.split(",") if v.strip()}
    by_id = {v.variant_id: v for v in VARIANTS}
    missing = sorted(v for v in wanted if v not in by_id)
    if missing:
        raise ValueError(f"Variantes desconocidas: {missing}")
    return [v for v in VARIANTS if v.variant_id in wanted]


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_ablation_db_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                out_dir TEXT NOT NULL,
                stable_ref TEXT NOT NULL,
                legacy_ref TEXT NOT NULL,
                seeds_json TEXT NOT NULL,
                bootstrap_samples INTEGER NOT NULL,
                permutation_samples INTEGER NOT NULL,
                workload_scale REAL NOT NULL,
                classification TEXT NOT NULL,
                rationale TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS variants (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                label TEXT NOT NULL,
                bridge_mode TEXT NOT NULL,
                policy_ref TEXT NOT NULL,
                bridge_ref TEXT NOT NULL,
                metrics_ref TEXT NOT NULL,
                fhs REAL,
                sss REAL,
                ops REAL,
                common_pass_rate REAL,
                ordered_pass_rate REAL,
                functional_regression INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, variant_id)
            );

            CREATE TABLE IF NOT EXISTS effects (
                run_id TEXT NOT NULL,
                effect_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, effect_key)
            );

            CREATE TABLE IF NOT EXISTS inference_diagnostics (
                run_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS negative_controls (
                run_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sensitivity (
                run_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS seed_metrics (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, variant_id, seed)
            );

            CREATE TABLE IF NOT EXISTS score_breakdown (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                score TEXT NOT NULL,
                component TEXT NOT NULL,
                weight REAL,
                normalized_value REAL,
                contribution REAL,
                clipped INTEGER NOT NULL,
                PRIMARY KEY (run_id, variant_id, seed, score, component)
            );

            CREATE TABLE IF NOT EXISTS geometry_family_metrics (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                geometry_name TEXT NOT NULL,
                family_code TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_text TEXT,
                applicable INTEGER NOT NULL,
                PRIMARY KEY (run_id, variant_id, seed, geometry_name, family_code, metric_name)
            );

            CREATE TABLE IF NOT EXISTS reasoning_family_metrics (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                geometry_name TEXT NOT NULL,
                geometry_family TEXT NOT NULL,
                reasoning_family TEXT NOT NULL,
                applicable INTEGER NOT NULL,
                present INTEGER,
                position INTEGER,
                PRIMARY KEY (run_id, variant_id, seed, geometry_name, geometry_family, reasoning_family)
            );

            CREATE TABLE IF NOT EXISTS family_coverage (
                run_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                family_code TEXT NOT NULL,
                geometry_count INTEGER NOT NULL,
                expected_present INTEGER NOT NULL,
                is_missing INTEGER NOT NULL,
                notes TEXT,
                PRIMARY KEY (run_id, variant_id, seed, family_code)
            );

            CREATE TABLE IF NOT EXISTS chart_manifest (
                run_id TEXT NOT NULL,
                chart_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                series_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, chart_id)
            );
            """
        )


def persist_run_to_sqlite(
    *,
    db_path: Path,
    run_id: str,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    effects: dict[str, dict[str, Any]],
    inference_diagnostics: dict[str, Any],
    negative_controls: dict[str, Any],
    sensitivity: dict[str, Any],
    category: str,
    rationale: str,
    chart_manifest: list[dict[str, Any]],
    family_details_rows: list[dict[str, Any]],
    family_summary_rows: list[dict[str, Any]],
    family_diagnostics: dict[str, Any],
) -> None:
    ensure_ablation_db_schema(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, timestamp, out_dir, stable_ref, legacy_ref, seeds_json,
                bootstrap_samples, permutation_samples, workload_scale,
                classification, rationale, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                report["timestamp"],
                str(report.get("analysis", {}).get("out_dir", report.get("out_dir", ""))),
                str(report["stable_ref"]),
                str(report["legacy_ref"]),
                _json_dumps(report.get("seeds", [])),
                int(report.get("bootstrap_samples", 0)),
                int(report.get("permutation_samples", 0)),
                float(report.get("workload_scale", 1.0)),
                category,
                rationale,
                datetime.now().isoformat(),
            ),
        )

        for row in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO variants (
                    run_id, variant_id, label, bridge_mode, policy_ref, bridge_ref, metrics_ref,
                    fhs, sss, ops, common_pass_rate, ordered_pass_rate, functional_regression, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    row["variant_id"],
                    row["label"],
                    row["bridge_mode"],
                    row["policy_ref"],
                    row["bridge_ref"],
                    row["metrics_ref"],
                    float(row.get("FHS", float("nan"))),
                    float(row.get("SSS", float("nan"))),
                    float(row.get("OPS", float("nan"))),
                    float(row.get("common_pass_rate", float("nan"))),
                    float(row.get("ordered_pass_rate", float("nan"))),
                    1 if bool(row.get("functional_regression", False)) else 0,
                    _json_dumps(row),
                ),
            )

            for item in row.get("seed_metrics_rows", []):
                seed = int(item.get("seed", -1))
                if seed < 0:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO seed_metrics (run_id, variant_id, seed, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, row["variant_id"], seed, _json_dumps(item)),
                )

            for item in row.get("score_breakdown_rows", []):
                seed = int(item.get("seed", -1))
                if seed < 0:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO score_breakdown (
                        run_id, variant_id, seed, score, component, weight,
                        normalized_value, contribution, clipped
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        row["variant_id"],
                        seed,
                        str(item.get("score", "")),
                        str(item.get("component", "")),
                        float(item.get("weight", 0.0)),
                        float(item.get("normalized_value", 0.0)),
                        float(item.get("contribution", 0.0)),
                        1 if bool(item.get("clipped", False)) else 0,
                    ),
                )

        for effect_key, payload in effects.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO effects (run_id, effect_key, payload_json)
                VALUES (?, ?, ?)
                """,
                (run_id, effect_key, _json_dumps(payload)),
            )

        conn.execute(
            "INSERT OR REPLACE INTO inference_diagnostics (run_id, payload_json) VALUES (?, ?)",
            (run_id, _json_dumps(inference_diagnostics)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO negative_controls (run_id, payload_json) VALUES (?, ?)",
            (run_id, _json_dumps(negative_controls)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO sensitivity (run_id, payload_json) VALUES (?, ?)",
            (run_id, _json_dumps(sensitivity)),
        )

        for detail in family_details_rows:
            metric_value = detail.get("value")
            metric_text: str | None = None
            if metric_value is None or (
                isinstance(metric_value, float) and not math.isfinite(metric_value)
            ):
                metric_numeric = None
            elif isinstance(metric_value, (int, float)):
                metric_numeric = float(metric_value)
            else:
                metric_numeric = None
                metric_text = str(metric_value)

            if detail.get("family_type") == "geometry":
                conn.execute(
                    """
                    INSERT OR REPLACE INTO geometry_family_metrics (
                        run_id, variant_id, seed, geometry_name, family_code,
                        metric_name, metric_value, metric_text, applicable
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        detail["variant_id"],
                        int(detail["seed"]),
                        str(detail.get("geometry_name", "")),
                        str(detail.get("family_id", "")),
                        str(detail.get("metric", "")),
                        metric_numeric,
                        metric_text,
                        1 if bool(detail.get("applicable", True)) else 0,
                    ),
                )
            elif detail.get("family_type") == "reasoning":
                present_val = detail.get("present")
                position_val = detail.get("position")
                present_int: int | None
                if present_val is None:
                    present_int = None
                else:
                    present_int = 1 if bool(present_val) else 0
                position_int: int | None
                if position_val is None:
                    position_int = None
                else:
                    position_int = int(position_val)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO reasoning_family_metrics (
                        run_id, variant_id, seed, geometry_name, geometry_family,
                        reasoning_family, applicable, present, position
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        detail["variant_id"],
                        int(detail["seed"]),
                        str(detail.get("geometry_name", "")),
                        str(detail.get("geometry_family", "")),
                        str(detail.get("family_id", "")),
                        1 if bool(detail.get("applicable", True)) else 0,
                        present_int,
                        position_int,
                    ),
                )

        for row in family_summary_rows:
            if row.get("metric") != "coverage_geometry_count":
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO family_coverage (
                    run_id, variant_id, seed, family_code, geometry_count,
                    expected_present, is_missing, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(row["variant_id"]),
                    int(row["seed"]),
                    str(row["family_id"]),
                    int(row.get("mean", 0.0)),
                    1,
                    1 if float(row.get("mean", 0.0)) <= 0.0 else 0,
                    str(family_diagnostics.get("status", "ok")),
                ),
            )

        for chart in chart_manifest:
            conn.execute(
                """
                INSERT OR REPLACE INTO chart_manifest (
                    run_id, chart_id, relative_path, sha256, size_bytes, series_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(chart.get("chart_id", "")),
                    str(chart.get("relative_path", "")),
                    str(chart.get("sha256", "")),
                    int(chart.get("size_bytes", 0)),
                    _json_dumps(chart.get("series", {})),
                    str(chart.get("created_at", datetime.now().isoformat())),
                ),
            )


def parse_junit(xml_path: Path) -> dict[str, Any]:
    if not xml_path.exists():
        return {
            "tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "failures_by_case": [],
            "failures_by_file": {},
        }
    root = ET.parse(xml_path).getroot()
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    else:
        suites = [root]

    tests = failed = errors = skipped = 0
    failures_by_case: list[dict[str, str]] = []
    failures_by_file: dict[str, str] = {}
    for ts in suites:
        tests += int(ts.attrib.get("tests", 0))
        failed += int(ts.attrib.get("failures", 0))
        errors += int(ts.attrib.get("errors", 0))
        skipped += int(ts.attrib.get("skipped", 0))
        for tc in ts.findall("testcase"):
            fail_node = tc.find("failure")
            err_node = tc.find("error")
            if fail_node is None and err_node is None:
                continue
            node = fail_node if fail_node is not None else err_node
            classname = tc.attrib.get("classname", "unknown")
            name = tc.attrib.get("name", "<unknown>")
            msg = (node.attrib.get("message", "") if node is not None else "").strip()
            text = ((node.text or "") if node is not None else "").strip()
            first_line = (msg or text or "failure/error").splitlines()[0][:300]
            failures_by_case.append(
                {
                    "classname": classname,
                    "name": name,
                    "message": first_line,
                }
            )
            file_key = classname.replace(".", "/") + ".py"
            if file_key not in failures_by_file:
                failures_by_file[file_key] = f"{name}: {first_line}"

    passed = max(0, tests - failed - errors - skipped)
    return {
        "tests": tests,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "failures_by_case": failures_by_case,
        "failures_by_file": failures_by_file,
    }


def parse_time_rss_kb(stderr: str) -> int | None:
    marker = "Maximum resident set size (kbytes):"
    for line in stderr.splitlines():
        if marker in line:
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def extract_metrics_json(stdout: str) -> dict[str, Any] | None:
    prefix = "METRICS_JSON="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            payload = line[len(prefix) :]
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
    return None


def run_pytest_suite(
    *,
    suite_name: str,
    tests_expr: str,
    root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    xml_path = out_dir / f"{suite_name}.xml"
    log_path = out_dir / f"{suite_name}.log"
    cmd = (
        f"pytest -q {tests_expr} --durations=20 --tb=short "
        f"--junitxml {xml_path}"
    )
    proc = run_cmd(
        cmd,
        cwd=root,
        log_path=log_path,
        shell=True,
    )
    junit = parse_junit(xml_path)
    return {
        "suite": suite_name,
        "command": cmd,
        "returncode": proc.returncode,
        "xml_path": str(xml_path),
        "log_path": str(log_path),
        "junit": junit,
    }


def run_inline_python(
    *,
    code: str,
    root: Path,
    log_path: Path,
) -> tuple[int, dict[str, Any] | None]:
    cmd = "python - <<'PY'\n" + code + "\nPY\n"
    proc = run_cmd(cmd, cwd=root, log_path=log_path, shell=True)
    return proc.returncode, extract_metrics_json(proc.stdout)


def run_timed_python(
    *,
    code: str,
    root: Path,
    log_path: Path,
) -> dict[str, Any]:
    cmd = "/usr/bin/time -v python - <<'PY'\n" + code + "\nPY\n"
    proc = run_cmd(cmd, cwd=root, log_path=log_path, shell=True)
    return {
        "returncode": proc.returncode,
        "metrics": extract_metrics_json(proc.stdout),
        "rss_kb": parse_time_rss_kb(proc.stderr),
        "stdout_tail": proc.stdout[-800:],
        "stderr_tail": proc.stderr[-800:],
    }


def workload_w1_code(seed: int, scale: float) -> str:
    n_iter = max(50_000, int(1_000_000 * scale))
    batch = 200
    return textwrap.dedent(
        f"""\
        import json, random, time
        from runtime.reasoning.scheduler_meta.budgeting import compute_budget
        from runtime.reasoning.scheduler_meta.policy import select_sequence

        N = {n_iter}
        B = {batch}
        rng = random.Random({seed})

        lat_us = []
        total = 0
        t0_all = time.perf_counter()

        while total < N:
            batch = B if (N - total) >= B else (N - total)
            feats = []
            for _ in range(batch):
                feats.append({{
                    'uncertainty': rng.random(),
                    'contradiction_signal': rng.random(),
                    'continuity_recent': rng.random(),
                    'edge_pressure': rng.random(),
                    'causal_risk': rng.random(),
                    'symbolic_regularity': rng.random(),
                    'law_fit_signal': rng.random(),
                }})
            t0 = time.perf_counter()
            for f in feats:
                b = compute_budget(f)
                select_sequence(features=f, budget=b, allow_experimental=True)
            dt = time.perf_counter() - t0
            lat_us.append((dt * 1e6) / batch)
            total += batch

        elapsed = time.perf_counter() - t0_all
        xs = sorted(lat_us)
        def q(p):
            i = int(p * (len(xs) - 1))
            return xs[i]
        m = {{
            'workload': 'W1',
            'seed': {seed},
            'iterations': N,
            'elapsed_s': elapsed,
            'iters_per_sec': N / elapsed,
            'p50_us': q(0.50),
            'p95_us': q(0.95),
        }}
        print('METRICS_JSON=' + json.dumps(m))
        """
    )


def workload_w2_code(seed: int, scale: float) -> str:
    n_iter = max(20_000, int(100_000 * scale))
    batch = 50
    return textwrap.dedent(
        f"""\
        import json, random, time
        from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler

        N = {n_iter}
        B = {batch}
        rng = random.Random({seed})
        sched = MetaScheduler(mode='adaptive')

        lat_us = []
        total = 0
        t0_all = time.perf_counter()
        while total < N:
            batch = B if (N - total) >= B else (N - total)
            ctxs = []
            for _ in range(batch):
                ctxs.append({{
                    'uncertainty': rng.random(),
                    'contradiction_signal': rng.random(),
                    'continuity_recent': rng.random(),
                    'edge_pressure': rng.random(),
                    'counterfactual_gap': rng.uniform(-1, 1),
                    'symbolic_regularity': rng.random(),
                    'law_fit_signal': rng.random(),
                }})
            t0 = time.perf_counter()
            for c in ctxs:
                sched.run(c)
            dt = time.perf_counter() - t0
            lat_us.append((dt * 1e6) / batch)
            total += batch

        elapsed = time.perf_counter() - t0_all
        xs = sorted(lat_us)
        def q(p):
            i = int(p * (len(xs) - 1))
            return xs[i]
        m = {{
            'workload': 'W2',
            'seed': {seed},
            'iterations': N,
            'elapsed_s': elapsed,
            'iters_per_sec': N / elapsed,
            'p50_us': q(0.50),
            'p95_us': q(0.95),
        }}
        print('METRICS_JSON=' + json.dumps(m))
        """
    )


def workload_w3_fractal_code(seed: int, scale: float) -> str:
    n_iter = max(30_000, int(120_000 * scale))
    batch = 40
    return textwrap.dedent(
        f"""\
        import json, random, time
        import numpy as np
        from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
        from tests.reasoning_stress.fractal_geometries import (
            FractalFamily,
            GeometricParameters,
            MetaParameters,
            generate_sierpinski_triangle,
            generate_cantor_carpet,
            generate_menger_sponge,
            generate_fractal_tree,
            generate_lorenz_attractor,
            generate_henon_attractor,
            generate_mandelbrot_set,
            generate_game_of_life_pattern,
            generate_scale_free_graph,
            generate_wavelet_decomposition,
            generate_kd_tree_partition,
            generate_fractional_brownian_motion,
            estimate_fractal_dimension_boxcount,
        )
        from tests.reasoning_stress.test_geometry_catalog import map_geometry_to_scheduler_features

        rng = random.Random({seed})
        np.random.seed({seed})
        meta = MetaParameters(lambda_rig=0.4, target_dimension=1.5)

        corpus = []
        skipped = []

        def _coerce_points(raw):
            if raw is None:
                return None, 'none'
            if isinstance(raw, tuple):
                if not raw:
                    return None, 'empty_tuple'
                raw = raw[0]
            arr = np.asarray(raw)
            if arr.size == 0:
                return None, 'empty'
            if arr.ndim == 1:
                arr = np.column_stack([np.linspace(0, 1, len(arr)), arr])
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.ndim != 2:
                return None, f'invalid_ndim_{{arr.ndim}}'
            finite_rows = np.isfinite(arr).all(axis=1)
            arr = arr[finite_rows]
            if len(arr) == 0:
                return None, 'all_non_finite'
            return arr.astype(float), ''

        gens = [
            ('triangular', lambda s: generate_sierpinski_triangle(GeometricParameters(family=FractalFamily.TRIANGULAR, depth=5, resolution=64, seed=s))),
            ('carpet', lambda s: generate_cantor_carpet(GeometricParameters(family=FractalFamily.CARPET, depth=3, grid_size=(3, 3), seed=s))),
            ('menger', lambda s: generate_menger_sponge(GeometricParameters(family=FractalFamily.VOLUMETRIC_3D, depth=2, seed=s))),
            ('tree', lambda s: generate_fractal_tree(GeometricParameters(family=FractalFamily.BRANCHING, branching_factor=2, depth=5, seed=s))[0]),
            ('lorenz', lambda s: generate_lorenz_attractor(GeometricParameters(family=FractalFamily.CONTINUOUS_ATTRACTOR, trajectory_length=2000, seed=s))[::8]),
            ('henon', lambda s: generate_henon_attractor(GeometricParameters(family=FractalFamily.DISCRETE_ATTRACTOR, trajectory_length=2500, seed=s))),
            ('mandelbrot', lambda s: generate_mandelbrot_set(GeometricParameters(family=FractalFamily.COMPLEX_PLANE, resolution=72, depth=50, seed=s))),
            ('life', lambda s: generate_game_of_life_pattern(GeometricParameters(family=FractalFamily.CELLULAR_AUTOMATA, grid_size=(48, 48), depth=25, seed=s))),
            ('scale_free', lambda s: generate_scale_free_graph(GeometricParameters(family=FractalFamily.FRACTAL_GRAPH, grid_size=(60,), branching_factor=3, seed=s))[0]),
            ('wavelet', lambda s: np.vstack([np.column_stack([np.linspace(0, 1, len(c)), c / (np.max(np.abs(c)) + 1e-10)]) for _, c in generate_wavelet_decomposition(GeometricParameters(family=FractalFamily.WAVELET, depth=3, resolution=128, seed=s)).items() if len(c) > 0])),
            ('kd_tree', lambda s: generate_kd_tree_partition(GeometricParameters(family=FractalFamily.PARTITION, depth=6, seed=s))),
            ('fbm', lambda s: generate_fractional_brownian_motion(GeometricParameters(family=FractalFamily.STOCHASTIC, trajectory_length=2048, hurst_exponent=0.7, seed=s))),
        ]

        for i, (name, fn) in enumerate(gens):
            try:
                raw = fn({seed} + i)
            except Exception as exc:
                skipped.append({{'name': name, 'reason': f'generator_error:{{exc.__class__.__name__}}'}})
                continue
            pts, reason = _coerce_points(raw)
            if pts is None:
                skipped.append({{'name': name, 'reason': reason}})
                continue
            try:
                fd, _ = estimate_fractal_dimension_boxcount(pts)
            except Exception as exc:
                skipped.append({{'name': name, 'reason': f'fd_error:{{exc.__class__.__name__}}'}})
                continue
            features = map_geometry_to_scheduler_features(pts, fd, meta)
            corpus.append(features)

        if not corpus:
            raise RuntimeError('empty fractal corpus after normalization')

        N = {n_iter}
        B = {batch}
        sched = MetaScheduler(mode='adaptive')
        lat_us = []
        total = 0
        t0_all = time.perf_counter()
        while total < N:
            batch = B if (N - total) >= B else (N - total)
            t0 = time.perf_counter()
            for j in range(batch):
                f = corpus[(total + j) % len(corpus)]
                ctx = dict(f)
                ctx['counterfactual_gap'] = rng.uniform(-1, 1)
                sched.run(ctx)
            dt = time.perf_counter() - t0
            lat_us.append((dt * 1e6) / batch)
            total += batch

        elapsed = time.perf_counter() - t0_all
        xs = sorted(lat_us)
        def q(p):
            i = int(p * (len(xs) - 1))
            return xs[i]
        m = {{
            'workload': 'W3',
            'seed': {seed},
            'iterations': N,
            'elapsed_s': elapsed,
            'iters_per_sec': N / elapsed,
            'p50_us': q(0.50),
            'p95_us': q(0.95),
            'corpus_size': len(corpus),
            'skipped_sources': skipped,
        }}
        print('METRICS_JSON=' + json.dumps(m))
        """
    )


def workload_w3_nonfractal_code(seed: int, scale: float) -> str:
    n_iter = max(30_000, int(120_000 * scale))
    batch = 40
    return textwrap.dedent(
        f"""\
        import json, random, time
        from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler

        rng = random.Random({seed})
        N = {n_iter}
        B = {batch}
        corpus_size = 2048

        corpus = []
        for i in range(corpus_size):
            phase = i % 4
            if phase == 0:
                feat = {{
                    'uncertainty': rng.uniform(0.15, 0.45),
                    'contradiction_signal': rng.uniform(0.0, 0.4),
                    'continuity_recent': rng.uniform(0.75, 1.0),
                    'edge_pressure': rng.uniform(0.0, 0.65),
                    'causal_risk': rng.uniform(0.0, 0.45),
                    'symbolic_regularity': rng.uniform(0.0, 0.35),
                    'law_fit_signal': rng.uniform(0.0, 0.35),
                }}
            elif phase == 1:
                feat = {{
                    'uncertainty': rng.uniform(0.55, 0.8),
                    'contradiction_signal': rng.uniform(0.45, 0.75),
                    'continuity_recent': rng.uniform(0.35, 0.8),
                    'edge_pressure': rng.uniform(0.68, 0.9),
                    'causal_risk': rng.uniform(0.45, 0.8),
                    'symbolic_regularity': rng.uniform(0.35, 0.7),
                    'law_fit_signal': rng.uniform(0.35, 0.7),
                }}
            elif phase == 2:
                feat = {{
                    'uncertainty': rng.uniform(0.58, 0.62),
                    'contradiction_signal': rng.uniform(0.42, 0.48),
                    'continuity_recent': rng.uniform(0.65, 0.9),
                    'edge_pressure': rng.uniform(0.67, 0.73),
                    'causal_risk': rng.uniform(0.48, 0.52),
                    'symbolic_regularity': rng.uniform(0.38, 0.42),
                    'law_fit_signal': rng.uniform(0.38, 0.42),
                }}
            else:
                feat = {{
                    'uncertainty': rng.random(),
                    'contradiction_signal': rng.random(),
                    'continuity_recent': rng.random(),
                    'edge_pressure': rng.random(),
                    'causal_risk': rng.random(),
                    'symbolic_regularity': rng.random(),
                    'law_fit_signal': rng.random(),
                }}
            corpus.append(feat)

        sched = MetaScheduler(mode='adaptive')
        lat_us = []
        total = 0
        t0_all = time.perf_counter()
        while total < N:
            batch = B if (N - total) >= B else (N - total)
            t0 = time.perf_counter()
            for j in range(batch):
                ctx = dict(corpus[(total + j) % corpus_size])
                ctx['counterfactual_gap'] = rng.uniform(-1, 1)
                sched.run(ctx)
            dt = time.perf_counter() - t0
            lat_us.append((dt * 1e6) / batch)
            total += batch

        elapsed = time.perf_counter() - t0_all
        xs = sorted(lat_us)
        def q(p):
            i = int(p * (len(xs) - 1))
            return xs[i]
        m = {{
            'workload': 'W3',
            'seed': {seed},
            'iterations': N,
            'elapsed_s': elapsed,
            'iters_per_sec': N / elapsed,
            'p50_us': q(0.50),
            'p95_us': q(0.95),
            'corpus_size': corpus_size,
            'source': 'nonfractal_profile',
        }}
        print('METRICS_JSON=' + json.dumps(m))
        """
    )


def functional_probe_code(seed: int) -> str:
    return textwrap.dedent(
        f"""\
        import json
        import random
        from runtime.reasoning.scheduler_meta.budgeting import compute_budget
        from runtime.reasoning.scheduler_meta.policy import select_sequence

        rng = random.Random({seed})

        def run_seq(features):
            b = compute_budget(features)
            seq, _, _ = select_sequence(features=features, budget=b, allow_experimental=True)
            return seq, b

        subthreshold_checks = [
            ("heur", {{
                "uncertainty": 0.25,
                "contradiction_signal": 0.0,
                "continuity_recent": 1.0,
                "edge_pressure": 0.699,
                "causal_risk": 0.0,
                "symbolic_regularity": 0.0,
                "law_fit_signal": 0.0,
            }}),
            ("dia_adv", {{
                "uncertainty": 0.25,
                "contradiction_signal": 0.449,
                "continuity_recent": 1.0,
                "edge_pressure": 0.0,
                "causal_risk": 0.0,
                "symbolic_regularity": 0.0,
                "law_fit_signal": 0.0,
            }}),
            ("fal_guard", {{
                "uncertainty": 0.25,
                "contradiction_signal": 0.449,
                "continuity_recent": 1.0,
                "edge_pressure": 0.0,
                "causal_risk": 0.0,
                "symbolic_regularity": 0.0,
                "law_fit_signal": 0.0,
            }}),
            ("eml_sr", {{
                "uncertainty": 0.25,
                "contradiction_signal": 0.0,
                "continuity_recent": 1.0,
                "edge_pressure": 0.0,
                "causal_risk": 0.0,
                "symbolic_regularity": 0.399,
                "law_fit_signal": 0.399,
            }}),
        ]

        false_subthreshold = 0
        activated_under_threshold = []
        for fam, features in subthreshold_checks:
            seq, _ = run_seq(features)
            if fam in seq:
                false_subthreshold += 1
                activated_under_threshold.append(fam)

        missing_cau = 0
        missing_ctf = 0
        total = 0
        for _ in range(500):
            features = {{
                "uncertainty": rng.random(),
                "contradiction_signal": rng.random(),
                "continuity_recent": rng.random(),
                "edge_pressure": rng.random(),
                "causal_risk": rng.random(),
                "symbolic_regularity": rng.random(),
                "law_fit_signal": rng.random(),
            }}
            seq, _ = run_seq(features)
            total += 1
            if "cau" not in seq:
                missing_cau += 1
            if "ctf" not in seq:
                missing_ctf += 1

        m = {{
            "seed": {seed},
            "false_subthreshold_count": false_subthreshold,
            "false_subthreshold_families": activated_under_threshold,
            "missing_cau_count": missing_cau,
            "missing_ctf_count": missing_ctf,
            "missing_core_rate": (missing_cau + missing_ctf) / max(1, 2 * total),
        }}
        print("METRICS_JSON=" + json.dumps(m))
        """
    )


def structural_probe_code(seed: int) -> str:
    return textwrap.dedent(
        f"""\
        import json
        import random
        import numpy as np
        from tests.reasoning_stress.fractal_utils import (
            measure_multiscale_boundary,
            estimate_box_counting_dimension,
            measure_temporal_cascades,
            measure_activation_avalanches,
        )

        random.seed({seed})
        np.random.seed({seed})

        boundaries = [
            ("edge_pressure", 0.7, "heur"),
            ("contradiction_signal", 0.45, "dia_adv"),
            ("contradiction_signal", 0.45, "fal_guard"),
            ("symbolic_regularity", 0.4, "eml_sr"),
            ("law_fit_signal", 0.4, "eml_sr"),
            ("uncertainty", 0.6, None),
            ("causal_risk", 0.5, None),
        ]

        mb = []
        for feat, thr, fam in boundaries:
            r = measure_multiscale_boundary(feature_name=feat, threshold=thr, family_name=fam)
            mb.append({{
                "converges": bool(r.converges),
                "roughness": float(r.roughness_exponent),
                "discipline": str(r.discipline),
                "convergence_rate": float(r.convergence_rate),
            }})

        box_specs = [
            ("contradiction_signal", "edge_pressure", "heur"),
            ("contradiction_signal", "uncertainty", "dia_adv"),
            ("symbolic_regularity", "law_fit_signal", "eml_sr"),
            ("edge_pressure", "causal_risk", "heur"),
        ]
        bc = []
        for x, y, fam in box_specs:
            r = estimate_box_counting_dimension(feature_x=x, feature_y=y, family_name=fam, n_samples=700)
            bc.append({{
                "dimension": float(r.fractal_dimension),
                "fit_quality": float(r.fit_quality),
                "interpretation": str(r.interpretation),
            }})

        temporal_specs = [
            ("uncertainty", "increasing"),
            ("uncertainty", "decreasing"),
            ("contradiction_signal", "increasing"),
            ("contradiction_signal", "decreasing"),
            ("edge_pressure", "increasing"),
            ("causal_risk", "increasing"),
        ]
        tc = []
        for feat, pat in temporal_specs:
            r = measure_temporal_cascades(feature_name=feat, perturbation_pattern=pat)
            tc.append({{
                "self_similar": bool(r.self_similar),
                "scale_error": float(r.scale_invariance_error),
                "fragility": float(r.temporal_memory_fragility),
            }})

        av = measure_activation_avalanches(n_trials=1000)
        hist_total = max(1, int(sum(av.size_histogram.values())))
        dominant_fraction = max(av.size_histogram.values()) / hist_total

        m = {{
            "seed": {seed},
            "multiscale": {{
                "convergent_ratio": float(sum(1 for r in mb if r["converges"]) / len(mb)),
                "pathological_count": int(sum(1 for r in mb if r["discipline"] == "pathological")),
                "mean_roughness": float(sum(r["roughness"] for r in mb) / len(mb)),
                "mean_convergence_rate": float(sum(r["convergence_rate"] for r in mb) / len(mb)),
            }},
            "box_counting": {{
                "mean_dimension": float(sum(r["dimension"] for r in bc) / len(bc)),
                "mean_fit_quality": float(sum(r["fit_quality"] for r in bc) / len(bc)),
                "pathological_count": int(sum(1 for r in bc if r["interpretation"] == "pathological")),
            }},
            "temporal": {{
                "self_similar_ratio": float(sum(1 for r in tc if r["self_similar"]) / len(tc)),
                "mean_scale_error": float(sum(r["scale_error"] for r in tc) / len(tc)),
                "mean_fragility": float(sum(r["fragility"] for r in tc) / len(tc)),
            }},
            "avalanche": {{
                "mean_size": float(av.mean_size),
                "max_size": int(av.max_size),
                "dominant_fraction": float(dominant_fraction),
                "criticality": str(av.criticalit_indicator),
                "heavy_tailed": bool(av.is_heavy_tailed),
            }},
        }}
        print("METRICS_JSON=" + json.dumps(m))
        """
    )


def family_probe_code(seed: int, bridge_mode: str) -> str:
    bridge_on = bridge_mode == "on"
    return textwrap.dedent(
        f"""\
        import json
        import numpy as np
        import math

        from runtime.reasoning.scheduler_meta.budgeting import compute_budget
        from runtime.reasoning.scheduler_meta.policy import select_sequence
        from tests.reasoning_stress.fractal_geometries import (
            FractalFamily,
            GeometricParameters,
            MetaParameters,
            generate_sierpinski_triangle,
            generate_cantor_carpet,
            generate_menger_sponge,
            generate_fractal_tree,
            generate_lorenz_attractor,
            generate_henon_attractor,
            generate_mandelbrot_set,
            generate_game_of_life_pattern,
            generate_scale_free_graph,
            generate_wavelet_decomposition,
            generate_kd_tree_partition,
            generate_fractional_brownian_motion,
            estimate_fractal_dimension_boxcount,
            compute_generalized_dimensions,
        )
        from tests.reasoning_stress.test_geometry_catalog import map_geometry_to_scheduler_features
        from tests.reasoning_stress.test_experiment2_atlas import (
            evaluate_selfsimilarity,
            evaluate_variety,
            evaluate_complexity,
            evaluate_memory,
            evaluate_edges,
            evaluate_communication,
            evaluate_vsh_evolution,
            evaluate_cognitive_alignment,
            evaluate_coherence,
        )

        seed = {seed}
        bridge_on = {str(bridge_on)}
        np.random.seed(seed)

        meta = MetaParameters(lambda_rig=0.4, target_dimension=1.5)
        reasoning_families = {json.dumps(REASONING_FAMILIES)}
        expected_families = {json.dumps(GEOMETRY_FAMILIES)}
        rows = []
        failures = []

        def _coerce_points(raw):
            if raw is None:
                return None
            if isinstance(raw, tuple):
                if not raw:
                    return None
                raw = raw[0]
            arr = np.asarray(raw, dtype=float)
            if arr.size == 0:
                return None
            if arr.ndim == 1:
                arr = np.column_stack([np.linspace(0, 1, len(arr)), arr])
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.ndim != 2:
                return None
            finite = np.isfinite(arr).all(axis=1)
            arr = arr[finite]
            if len(arr) == 0:
                return None
            return arr

        configs = [
            ("Sierpinski Triangle", FractalFamily.TRIANGULAR, lambda s: generate_sierpinski_triangle(GeometricParameters(family=FractalFamily.TRIANGULAR, depth=5, resolution=64, seed=s))),
            ("Cantor Carpet", FractalFamily.CARPET, lambda s: generate_cantor_carpet(GeometricParameters(family=FractalFamily.CARPET, grid_size=(3, 3), depth=3, seed=s))),
            ("Menger Sponge", FractalFamily.VOLUMETRIC_3D, lambda s: generate_menger_sponge(GeometricParameters(family=FractalFamily.VOLUMETRIC_3D, depth=3, seed=s))),
            ("Binary Tree", FractalFamily.BRANCHING, lambda s: generate_fractal_tree(GeometricParameters(family=FractalFamily.BRANCHING, branching_factor=2, depth=5, seed=s))[0]),
            ("Lorenz Attractor", FractalFamily.CONTINUOUS_ATTRACTOR, lambda s: generate_lorenz_attractor(GeometricParameters(family=FractalFamily.CONTINUOUS_ATTRACTOR, trajectory_length=2000, seed=s))[::8]),
            ("Henon Map", FractalFamily.DISCRETE_ATTRACTOR, lambda s: generate_henon_attractor(GeometricParameters(family=FractalFamily.DISCRETE_ATTRACTOR, trajectory_length=3000, seed=s))),
            ("Mandelbrot Set", FractalFamily.COMPLEX_PLANE, lambda s: generate_mandelbrot_set(GeometricParameters(family=FractalFamily.COMPLEX_PLANE, resolution=80, depth=60, seed=s))),
            ("Game of Life", FractalFamily.CELLULAR_AUTOMATA, lambda s: generate_game_of_life_pattern(GeometricParameters(family=FractalFamily.CELLULAR_AUTOMATA, grid_size=(64, 64), depth=30, seed=s))),
            ("Scale-Free Graph", FractalFamily.FRACTAL_GRAPH, lambda s: generate_scale_free_graph(GeometricParameters(family=FractalFamily.FRACTAL_GRAPH, grid_size=(60,), branching_factor=3, seed=s))[0]),
            ("Wavelet Decomposition", FractalFamily.WAVELET, lambda s: np.vstack([np.column_stack([np.linspace(0, 1, len(c)), c / (np.max(np.abs(c)) + 1e-10)]) for _, c in generate_wavelet_decomposition(GeometricParameters(family=FractalFamily.WAVELET, depth=3, resolution=128, seed=s)).items() if len(c) > 0])),
            ("KD-Tree Partition", FractalFamily.PARTITION, lambda s: generate_kd_tree_partition(GeometricParameters(family=FractalFamily.PARTITION, depth=5, resolution=100, seed=s))[0]),
            ("Fractional Brownian", FractalFamily.STOCHASTIC, lambda s: (lambda fbm: np.column_stack([np.arange(len(fbm)) / max(1, len(fbm)), fbm / (np.max(np.abs(fbm)) + 1e-10)]))(generate_fractional_brownian_motion(GeometricParameters(family=FractalFamily.STOCHASTIC, hurst_exponent=0.7, resolution=256, seed=s)))),
        ]

        for idx, (name, family, fn) in enumerate(configs):
            try:
                pts = _coerce_points(fn(seed + idx))
                if pts is None:
                    failures.append({{"name": name, "family": family.value, "error": "empty_or_invalid_points"}})
                    continue
                fd, r2 = estimate_fractal_dimension_boxcount(pts)
                d_q = compute_generalized_dimensions(pts, q_values=[0.0, 1.0, 2.0])
                features = map_geometry_to_scheduler_features(pts, fd, meta)

                f1 = evaluate_selfsimilarity(pts, fd)
                f2 = evaluate_variety(family)
                f3 = evaluate_complexity(pts, fd)
                f4 = evaluate_memory(family)
                f5 = evaluate_edges(family)
                f6 = evaluate_communication(family, fd)
                f7 = evaluate_vsh_evolution(family)
                f8 = evaluate_cognitive_alignment(fd, meta.target_dimension)
                f9 = evaluate_coherence(family)
                atlas_scores = {{
                    "F1": float(f1["score"]),
                    "F2": float(f2["score"]),
                    "F3": float(f3["score"]),
                    "F4": float(f4["score"]),
                    "F5": float(f5["score"]),
                    "F6": float(f6["score"]),
                    "F7": float(f7["score"]),
                    "F8": float(f8["score"]),
                    "F9": float(f9["score"]),
                }}
                atlas_overall = float(np.mean(list(atlas_scores.values())))

                if bridge_on:
                    budget = compute_budget(features)
                    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)
                    applicable = True
                else:
                    budget = {{}}
                    sequence = []
                    applicable = False

                presence = {{}}
                position = {{}}
                for fam in reasoning_families:
                    if applicable:
                        presence[fam] = bool(fam in sequence)
                        position[fam] = int(sequence.index(fam)) if fam in sequence else None
                    else:
                        presence[fam] = None
                        position[fam] = None

                rows.append({{
                    "geometry_name": name,
                    "geometry_family": family.value,
                    "fd": float(fd),
                    "r2": float(r2),
                    "d_q_0": float(d_q.get(0.0, fd)),
                    "d_q_1": float(d_q.get(1.0, fd)),
                    "d_q_2": float(d_q.get(2.0, fd)),
                    "atlas_scores": atlas_scores,
                    "atlas_overall": atlas_overall,
                    "scheduler_features": {{k: float(v) for k, v in features.items()}},
                    "max_steps": float(budget.get("max_steps", float("nan"))) if applicable else None,
                    "sequence_length": int(len(sequence)) if applicable else None,
                    "reasoning_applicable": applicable,
                    "reasoning_presence": presence,
                    "reasoning_position": position,
                }})
            except Exception as exc:
                failures.append({{"name": name, "family": family.value, "error": f"{{exc.__class__.__name__}}:{{exc}}"}})

        observed = sorted({{r["geometry_family"] for r in rows}})
        missing = sorted([f for f in expected_families if f not in observed])

        out = {{
            "seed": seed,
            "bridge_mode": "on" if bridge_on else "off",
            "rows": rows,
            "coverage": {{
                "expected_families": expected_families,
                "observed_families": observed,
                "missing_families": missing,
                "n_expected": len(expected_families),
                "n_observed": len(observed),
            }},
            "failures": failures,
        }}
        print("METRICS_JSON=" + json.dumps(out))
        """
    )


def summarize_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in repeats if r.get("metrics")]
    if not ok:
        return {"ok_runs": 0}

    keys = ["elapsed_s", "iters_per_sec", "p50_us", "p95_us"]
    out: dict[str, Any] = {"ok_runs": len(ok)}
    for key in keys:
        vals = [float(r["metrics"][key]) for r in ok]
        out[key] = {
            "mean": statistics.mean(vals),
            "stdev": statistics.pstdev(vals),
            "values": vals,
        }
    rss_vals = [int(r["rss_kb"]) for r in ok if r.get("rss_kb") is not None]
    if rss_vals:
        out["rss_kb"] = {
            "mean": statistics.mean(rss_vals),
            "stdev": statistics.pstdev(rss_vals),
            "values": [float(v) for v in rss_vals],
        }
    return out


def summarize_probe_runs(probe_runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [p for p in probe_runs if p]
    if not ok:
        return {"ok_runs": 0}

    seed_details = sorted(
        [
            {
                "seed": int(p.get("seed", -1)),
                "false_subthreshold_count": float(p["false_subthreshold_count"]),
                "missing_core_rate": float(p["missing_core_rate"]),
                "missing_cau_count": float(p["missing_cau_count"]),
                "missing_ctf_count": float(p["missing_ctf_count"]),
            }
            for p in ok
        ],
        key=lambda x: x["seed"],
    )

    false_sub = [float(p["false_subthreshold_count"]) for p in ok]
    missing_core = [float(p["missing_core_rate"]) for p in ok]
    missing_cau = [float(p["missing_cau_count"]) for p in ok]
    missing_ctf = [float(p["missing_ctf_count"]) for p in ok]
    return {
        "ok_runs": len(ok),
        "false_subthreshold_count": {
            "mean": statistics.mean(false_sub),
            "stdev": statistics.pstdev(false_sub),
            "values": false_sub,
        },
        "missing_core_rate": {
            "mean": statistics.mean(missing_core),
            "stdev": statistics.pstdev(missing_core),
            "values": missing_core,
        },
        "missing_cau_count": {
            "mean": statistics.mean(missing_cau),
            "stdev": statistics.pstdev(missing_cau),
            "values": missing_cau,
        },
        "missing_ctf_count": {
            "mean": statistics.mean(missing_ctf),
            "stdev": statistics.pstdev(missing_ctf),
            "values": missing_ctf,
        },
        "seed_details": seed_details,
    }


def compute_sss_from_components(
    components: dict[str, float],
    *,
    excluded: set[str] | None = None,
) -> tuple[float, dict[str, float]]:
    excluded = excluded or set()
    active = [k for k in SSS_COMPONENT_WEIGHTS if k not in excluded]
    if not active:
        return float("nan"), {}
    total_w = sum(SSS_COMPONENT_WEIGHTS[k] for k in active)
    contributions: dict[str, float] = {}
    total = 0.0
    for name in active:
        w_norm = SSS_COMPONENT_WEIGHTS[name] / total_w
        val = clip(float(components.get(name, 0.0)), 0.0, 1.0)
        c = 100.0 * w_norm * val
        contributions[name] = c
        total += c
    return clip(total, 0.0, 100.0), contributions


def summarize_structural_runs(struct_runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [p for p in struct_runs if p]
    if not ok:
        return {"ok_runs": 0}

    def collect(path: tuple[str, ...]) -> list[float]:
        vals: list[float] = []
        for item in ok:
            node: Any = item
            for key in path:
                node = node[key]
            vals.append(float(node))
        return vals

    convergent_ratio = collect(("multiscale", "convergent_ratio"))
    mb_pathological = collect(("multiscale", "pathological_count"))
    roughness = collect(("multiscale", "mean_roughness"))
    conv_rate = collect(("multiscale", "mean_convergence_rate"))
    mean_dim = collect(("box_counting", "mean_dimension"))
    fit_q = collect(("box_counting", "mean_fit_quality"))
    bc_pathological = collect(("box_counting", "pathological_count"))
    self_sim = collect(("temporal", "self_similar_ratio"))
    scale_error = collect(("temporal", "mean_scale_error"))
    fragility = collect(("temporal", "mean_fragility"))
    avalanche_mean = collect(("avalanche", "mean_size"))
    dominant_frac = collect(("avalanche", "dominant_fraction"))

    seed_details: list[dict[str, Any]] = []
    sss_runs: list[float] = []
    for i, run in enumerate(ok):
        seed = int(run.get("seed", i))
        components = {
            "convergent_ratio": clip(convergent_ratio[i], 0.0, 1.0),
            "non_path_mb": 1.0 - clip(mb_pathological[i] / 7.0, 0.0, 1.0),
            "non_path_bc": 1.0 - clip(bc_pathological[i] / 4.0, 0.0, 1.0),
            "self_similar": clip(self_sim[i], 0.0, 1.0),
            "avalanche_diversity": clip(1.0 - dominant_frac[i], 0.0, 1.0),
            "temporal_stability": clip(1.0 - clip(scale_error[i], 0.0, 1.0), 0.0, 1.0),
        }
        sss, contrib = compute_sss_from_components(components)
        sss_runs.append(sss)
        seed_details.append(
            {
                "seed": seed,
                "raw": {
                    "convergent_ratio": convergent_ratio[i],
                    "pathological_multiscale_count": mb_pathological[i],
                    "pathological_box_counting_count": bc_pathological[i],
                    "self_similar_ratio": self_sim[i],
                    "dominant_fraction": dominant_frac[i],
                    "temporal_scale_error": scale_error[i],
                    "mean_dimension": mean_dim[i],
                    "mean_fit_quality": fit_q[i],
                    "mean_roughness": roughness[i],
                    "mean_convergence_rate": conv_rate[i],
                    "mean_fragility": fragility[i],
                    "mean_avalanche_size": avalanche_mean[i],
                },
                "normalized": components,
                "weights": dict(SSS_COMPONENT_WEIGHTS),
                "contributions": contrib,
                "SSS": sss,
            }
        )

    seed_details.sort(key=lambda x: x["seed"])

    return {
        "ok_runs": len(ok),
        "convergent_ratio": {"mean": statistics.mean(convergent_ratio), "values": convergent_ratio},
        "pathological_multiscale_count": {"mean": statistics.mean(mb_pathological), "values": mb_pathological},
        "mean_roughness": {"mean": statistics.mean(roughness), "values": roughness},
        "mean_convergence_rate": {"mean": statistics.mean(conv_rate), "values": conv_rate},
        "box_mean_dimension": {"mean": statistics.mean(mean_dim), "values": mean_dim},
        "box_mean_fit_quality": {"mean": statistics.mean(fit_q), "values": fit_q},
        "box_pathological_count": {"mean": statistics.mean(bc_pathological), "values": bc_pathological},
        "temporal_self_similar_ratio": {"mean": statistics.mean(self_sim), "values": self_sim},
        "temporal_scale_error": {"mean": statistics.mean(scale_error), "values": scale_error},
        "temporal_fragility": {"mean": statistics.mean(fragility), "values": fragility},
        "avalanche_mean_size": {"mean": statistics.mean(avalanche_mean), "values": avalanche_mean},
        "avalanche_dominant_fraction": {"mean": statistics.mean(dominant_frac), "values": dominant_frac},
        "sss_runs": sss_runs,
        "sss": {"mean": statistics.mean(sss_runs), "stdev": statistics.pstdev(sss_runs), "values": sss_runs},
        "seed_details": seed_details,
    }


def summarize_family_runs(
    family_runs: list[dict[str, Any]],
    *,
    bridge_mode: str,
) -> dict[str, Any]:
    ok_runs = [r for r in family_runs if isinstance(r, dict) and r.get("rows")]
    if not ok_runs:
        return {
            "ok_runs": 0,
            "family_details_rows": [],
            "family_summary_rows": [],
            "family_diagnostics": {
                "status": "no_data",
                "expected_geometry_families": GEOMETRY_FAMILIES,
                "observed_geometry_families": [],
                "missing_geometry_families": GEOMETRY_FAMILIES,
                "bridge_mode": bridge_mode,
            },
        }

    details_rows: list[dict[str, Any]] = []
    summary_acc: dict[tuple[str, str, str], list[float]] = {}
    coverage_per_seed: list[dict[str, Any]] = []
    observed_global: set[str] = set()
    failures: list[dict[str, Any]] = []

    geometry_metrics = [
        "fd",
        "r2",
        "d_q_0",
        "d_q_1",
        "d_q_2",
        "atlas_overall",
        "max_steps",
        "sequence_length",
    ]
    feature_metrics = [
        "uncertainty",
        "contradiction_signal",
        "continuity_recent",
        "edge_pressure",
        "causal_risk",
        "symbolic_regularity",
        "law_fit_signal",
    ]
    atlas_metrics = [f"F{i}" for i in range(1, 10)]

    for run in ok_runs:
        seed = int(run.get("seed", -1))
        rows = run.get("rows", [])
        coverage = run.get("coverage", {})
        observed = coverage.get("observed_families", [])
        missing = coverage.get("missing_families", [])
        observed_global.update(observed)
        failures.extend(run.get("failures", []))
        coverage_per_seed.append(
            {
                "seed": seed,
                "observed_families": observed,
                "missing_families": missing,
                "n_observed": int(coverage.get("n_observed", 0)),
            }
        )

        present_counter: dict[str, float] = {fam: 0.0 for fam in REASONING_FAMILIES}
        applicable_counter: dict[str, float] = {fam: 0.0 for fam in REASONING_FAMILIES}
        position_values: dict[str, list[float]] = {fam: [] for fam in REASONING_FAMILIES}
        geometry_counter: dict[str, int] = {fam: 0 for fam in GEOMETRY_FAMILIES}

        for geom in rows:
            family_code = str(geom.get("geometry_family", ""))
            geometry_name = str(geom.get("geometry_name", ""))
            geometry_counter[family_code] = geometry_counter.get(family_code, 0) + 1
            applicable = bool(geom.get("reasoning_applicable", False))

            for metric in geometry_metrics:
                value = geom.get(metric)
                details_rows.append(
                    {
                        "seed": seed,
                        "family_type": "geometry",
                        "family_id": family_code,
                        "geometry_name": geometry_name,
                        "geometry_family": family_code,
                        "metric": metric,
                        "value": value,
                        "applicable": True,
                    }
                )
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    summary_acc.setdefault(("geometry", family_code, metric), []).append(float(value))

            atlas_scores = geom.get("atlas_scores", {})
            for metric in atlas_metrics:
                value = atlas_scores.get(metric)
                details_rows.append(
                    {
                        "seed": seed,
                        "family_type": "geometry",
                        "family_id": family_code,
                        "geometry_name": geometry_name,
                        "geometry_family": family_code,
                        "metric": metric,
                        "value": value,
                        "applicable": True,
                    }
                )
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    summary_acc.setdefault(("geometry", family_code, metric), []).append(float(value))

            features = geom.get("scheduler_features", {})
            for metric in feature_metrics:
                value = features.get(metric)
                details_rows.append(
                    {
                        "seed": seed,
                        "family_type": "geometry",
                        "family_id": family_code,
                        "geometry_name": geometry_name,
                        "geometry_family": family_code,
                        "metric": metric,
                        "value": value,
                        "applicable": True,
                    }
                )
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    summary_acc.setdefault(("geometry", family_code, metric), []).append(float(value))

            presence = geom.get("reasoning_presence", {})
            positions = geom.get("reasoning_position", {})
            for rfam in REASONING_FAMILIES:
                p_val = presence.get(rfam, None)
                pos_val = positions.get(rfam, None)
                if applicable:
                    applicable_counter[rfam] += 1.0
                    if p_val:
                        present_counter[rfam] += 1.0
                    if pos_val is not None:
                        position_values[rfam].append(float(pos_val))
                details_rows.append(
                    {
                        "seed": seed,
                        "family_type": "reasoning",
                        "family_id": rfam,
                        "geometry_name": geometry_name,
                        "geometry_family": family_code,
                        "metric": "present",
                        "value": (1.0 if p_val else 0.0) if p_val is not None else None,
                        "present": p_val,
                        "position": pos_val,
                        "applicable": applicable,
                    }
                )
                details_rows.append(
                    {
                        "seed": seed,
                        "family_type": "reasoning",
                        "family_id": rfam,
                        "geometry_name": geometry_name,
                        "geometry_family": family_code,
                        "metric": "position",
                        "value": float(pos_val) if pos_val is not None else None,
                        "present": p_val,
                        "position": pos_val,
                        "applicable": applicable,
                    }
                )

        for fam_code in GEOMETRY_FAMILIES:
            cnt = float(geometry_counter.get(fam_code, 0))
            summary_acc.setdefault(("geometry", fam_code, "coverage_geometry_count"), []).append(cnt)

        for rfam in REASONING_FAMILIES:
            denom = applicable_counter[rfam]
            if denom > 0.0:
                act = present_counter[rfam] / denom
                summary_acc.setdefault(("reasoning", rfam, "activation_rate"), []).append(float(act))
                if position_values[rfam]:
                    pos_mean = statistics.mean(position_values[rfam])
                    summary_acc.setdefault(("reasoning", rfam, "mean_position"), []).append(float(pos_mean))
            else:
                summary_acc.setdefault(("reasoning", rfam, "activation_rate"), []).append(float("nan"))
                summary_acc.setdefault(("reasoning", rfam, "mean_position"), []).append(float("nan"))

    summary_rows: list[dict[str, Any]] = []
    for (family_type, family_id, metric), vals in sorted(summary_acc.items()):
        clean = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        n = len(clean)
        if n == 0:
            mean = float("nan")
            stdev = float("nan")
        else:
            mean = float(statistics.mean(clean))
            stdev = float(statistics.pstdev(clean))
        summary_rows.append(
            {
                "family_type": family_type,
                "family_id": family_id,
                "metric": metric,
                "mean": mean,
                "stdev": stdev,
                "n": n,
                "seed": -1,
            }
        )

    for cov in coverage_per_seed:
        seed = int(cov["seed"])
        missing_set = set(cov.get("missing_families", []))
        for fam in GEOMETRY_FAMILIES:
            summary_rows.append(
                {
                    "family_type": "geometry",
                    "family_id": fam,
                    "metric": "coverage_geometry_count",
                    "mean": 0.0 if fam in missing_set else 1.0,
                    "stdev": 0.0,
                    "n": 1,
                    "seed": seed,
                }
            )

    missing_global = sorted([f for f in GEOMETRY_FAMILIES if f not in observed_global])
    diagnostics = {
        "status": "ok" if not missing_global else "partial_coverage",
        "bridge_mode": bridge_mode,
        "expected_geometry_families": GEOMETRY_FAMILIES,
        "observed_geometry_families": sorted(observed_global),
        "missing_geometry_families": missing_global,
        "ok_runs": len(ok_runs),
        "seed_coverage": coverage_per_seed,
        "failures": failures,
    }
    return {
        "ok_runs": len(ok_runs),
        "family_details_rows": details_rows,
        "family_summary_rows": summary_rows,
        "family_diagnostics": diagnostics,
    }


def clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def write_git_file_from_ref(repo_root: Path, ref: str, rel_path: str, dest_root: Path) -> None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    target = dest_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(proc.stdout, encoding="utf-8")


def materialize_variant_root(
    *,
    repo_root: Path,
    stable_ref: str,
    legacy_ref: str,
    variant: VariantSpec,
    variant_root: Path,
) -> None:
    if variant_root.exists():
        shutil.rmtree(variant_root)
    variant_root.mkdir(parents=True, exist_ok=True)

    archive_cmd = f"git archive --format=tar {stable_ref} | tar -x -C {variant_root}"
    proc = run_cmd(archive_cmd, cwd=repo_root, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"No se pudo materializar base estable en {variant_root}")

    refs = {"stable": stable_ref, "legacy": legacy_ref}
    policy_ref = refs[variant.policy_ref]
    bridge_ref = refs[variant.bridge_ref]
    metrics_ref = refs[variant.metrics_ref]

    write_git_file_from_ref(repo_root, policy_ref, POLICY_FILE, variant_root)
    write_git_file_from_ref(repo_root, bridge_ref, BRIDGE_FILE, variant_root)
    for rel in METRICS_FILES:
        write_git_file_from_ref(repo_root, metrics_ref, rel, variant_root)

    # Guard anti-sesgo: fix de pairwise forzado estable en TODAS las variantes.
    write_git_file_from_ref(repo_root, stable_ref, PAIRWISE_FIX_FILE, variant_root)


def _bench_by_seed(repeats: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for rep in repeats:
        metrics = rep.get("metrics")
        if not metrics:
            continue
        seed_raw = metrics.get("seed")
        if seed_raw is None:
            continue
        seed = int(seed_raw)
        out[seed] = {
            "iters_per_sec": float(metrics.get("iters_per_sec", float("nan"))),
            "p95_us": float(metrics.get("p95_us", float("nan"))),
            "rss_kb": float(rep["rss_kb"]) if rep.get("rss_kb") is not None else float("nan"),
        }
    return out


def _aligned_seed_vectors(base_map: dict[int, float], other_map: dict[int, float]) -> tuple[list[int], list[float], list[float]]:
    seeds = sorted(set(base_map.keys()).intersection(other_map.keys()))
    base_vals = [float(base_map[s]) for s in seeds if math.isfinite(float(base_map[s]))]
    other_vals = [float(other_map[s]) for s in seeds if math.isfinite(float(other_map[s]))]
    if len(base_vals) != len(other_vals):
        paired: list[tuple[int, float, float]] = []
        for s in seeds:
            b = float(base_map[s])
            o = float(other_map[s])
            if math.isfinite(b) and math.isfinite(o):
                paired.append((s, b, o))
        seeds = [p[0] for p in paired]
        base_vals = [p[1] for p in paired]
        other_vals = [p[2] for p in paired]
    return seeds, base_vals, other_vals


def _metric_effect_from_seed_maps(
    *,
    base_map: dict[int, float],
    other_map: dict[int, float],
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    larger_is_better: bool = True,
    min_n: int = 3,
) -> dict[str, Any]:
    seeds, base_vals, other_vals = _aligned_seed_vectors(base_map, other_map)
    delta = bootstrap_ci_delta_abs(base_vals, other_vals, n_boot=bootstrap_samples, seed=seed)
    g = hedges_g(base_vals, other_vals)
    cdelta = cliffs_delta(base_vals, other_vals, larger_is_better=larger_is_better)
    perm = permutation_test_mean_delta(
        base_vals,
        other_vals,
        n_perm=permutation_samples,
        seed=seed + 997,
    )
    enough_n = len(seeds) >= min_n
    delta_nan = not math.isfinite(float(delta.get("point", float("nan"))))
    ci_low = float(delta.get("ci95_low", float("nan")))
    ci_high = float(delta.get("ci95_high", float("nan")))
    ci_degenerate = bool(delta.get("ci95_degenerate", False))
    if math.isfinite(ci_low) and math.isfinite(ci_high) and abs(ci_high - ci_low) <= 1e-12:
        ci_degenerate = True
    diagnostics = {
        "n_pairs": len(seeds),
        "paired_seeds": seeds,
        "enough_n": enough_n,
        "min_n_required": min_n,
        "delta_has_nan": delta_nan,
        "ci_degenerate": ci_degenerate,
        "hedges_g_undefined": bool(g.get("undefined", True)),
        "hedges_g_reason": g.get("reason", "unknown"),
        "cliffs_has_nan": not math.isfinite(float(cdelta.get("value", float("nan")))),
        "permutation_has_nan": not math.isfinite(float(perm.get("p_two_sided", float("nan")))),
        "sign_consistent": bool(
            math.isfinite(float(delta.get("point", float("nan"))))
            and math.isfinite(float(cdelta.get("value", float("nan"))))
            and (
                (float(delta["point"]) == 0.0 and float(cdelta["value"]) == 0.0)
                or (float(delta["point"]) > 0 and float(cdelta["value"]) > 0)
                or (float(delta["point"]) < 0 and float(cdelta["value"]) < 0)
            )
        ),
    }
    return {
        "delta": delta,
        "hedges_g": g,
        "cliffs_delta": cdelta,
        "permutation": perm,
        "diagnostics": diagnostics,
    }


def calc_variant_scores(
    *,
    variant_report: dict[str, Any],
    baseline_report: dict[str, Any] | None,
    bootstrap_samples: int,
) -> dict[str, Any]:
    suites = variant_report.get("suites", {})
    common_total = 0
    common_passed = 0
    common_failed = 0
    common_errors = 0
    for key in ("common_regression", "common_stress"):
        junit = suites.get(key, {}).get("junit", {})
        common_total += int(junit.get("tests", 0))
        common_passed += int(junit.get("passed", 0))
        common_failed += int(junit.get("failed", 0))
        common_errors += int(junit.get("errors", 0))

    ordered_junit = suites.get("ordered_suite", {}).get("junit", {})
    ordered_total = int(ordered_junit.get("tests", 0))
    ordered_passed = int(ordered_junit.get("passed", 0))

    common_rate = (common_passed / common_total) if common_total else 0.0
    ordered_rate = (ordered_passed / ordered_total) if ordered_total else 0.0
    common_wilson = wilson_interval(common_passed, common_total)
    ordered_wilson = wilson_interval(ordered_passed, ordered_total)
    functional_regression = common_failed > 0 or common_errors > 0

    functional_probe = variant_report.get("functional_probe_summary", {})
    false_sub_mean = float(functional_probe.get("false_subthreshold_count", {}).get("mean", 0.0))
    missing_core_rate = float(functional_probe.get("missing_core_rate", {}).get("mean", 0.0))
    functional_seed_details = [
        d for d in functional_probe.get("seed_details", []) if int(d.get("seed", -1)) >= 0
    ]
    functional_seed_details.sort(key=lambda x: int(x["seed"]))

    fhs_by_seed: dict[int, float] = {}
    fhs_breakdown: list[dict[str, Any]] = []
    fhs_common_contrib = 100.0 * 0.55 * common_rate
    fhs_ordered_contrib = 100.0 * 0.25 * ordered_rate
    for d in functional_seed_details:
        seed = int(d["seed"])
        false_sub = float(d["false_subthreshold_count"])
        missing_core = float(d["missing_core_rate"])
        false_sub_norm = 1.0 - clip(false_sub / 4.0, 0.0, 1.0)
        false_sub_contrib = 100.0 * 0.20 * false_sub_norm
        penalty_missing = 35.0 * clip(missing_core * 8.0, 0.0, 1.0)
        penalty_reg = 30.0 if functional_regression else 0.0
        fhs_seed = clip(
            fhs_common_contrib + fhs_ordered_contrib + false_sub_contrib - penalty_missing - penalty_reg,
            0.0,
            100.0,
        )
        fhs_by_seed[seed] = fhs_seed
        fhs_breakdown.extend(
            [
                {
                    "seed": seed,
                    "score": "FHS",
                    "component": "common_pass_rate",
                    "weight": 0.55,
                    "normalized_value": common_rate,
                    "contribution": fhs_common_contrib,
                    "clipped": False,
                },
                {
                    "seed": seed,
                    "score": "FHS",
                    "component": "ordered_pass_rate",
                    "weight": 0.25,
                    "normalized_value": ordered_rate,
                    "contribution": fhs_ordered_contrib,
                    "clipped": False,
                },
                {
                    "seed": seed,
                    "score": "FHS",
                    "component": "subthreshold_discipline",
                    "weight": 0.20,
                    "normalized_value": false_sub_norm,
                    "contribution": false_sub_contrib,
                    "clipped": False,
                },
                {
                    "seed": seed,
                    "score": "FHS",
                    "component": "missing_core_penalty",
                    "weight": -35.0,
                    "normalized_value": clip(missing_core * 8.0, 0.0, 1.0),
                    "contribution": -penalty_missing,
                    "clipped": True,
                },
                {
                    "seed": seed,
                    "score": "FHS",
                    "component": "functional_regression_penalty",
                    "weight": -30.0,
                    "normalized_value": 1.0 if functional_regression else 0.0,
                    "contribution": -penalty_reg,
                    "clipped": False,
                },
            ]
        )

    fhs = statistics.mean(fhs_by_seed.values()) if fhs_by_seed else clip(
        100.0
        * (
            0.55 * common_rate
            + 0.25 * ordered_rate
            + 0.20 * (1.0 - clip(false_sub_mean / 4.0, 0.0, 1.0))
        )
        - (35.0 * clip(missing_core_rate * 8.0, 0.0, 1.0))
        - (30.0 if functional_regression else 0.0),
        0.0,
        100.0,
    )

    structural = variant_report.get("structural_probe_summary", {})
    dominant_fraction = float(structural.get("avalanche_dominant_fraction", {}).get("mean", 1.0))
    pathological_multiscale = float(structural.get("pathological_multiscale_count", {}).get("mean", 7.0))
    temporal_scale_error = float(structural.get("temporal_scale_error", {}).get("mean", 1.0))
    structural_seed_details = [
        d for d in structural.get("seed_details", []) if int(d.get("seed", -1)) >= 0
    ]
    structural_seed_details.sort(key=lambda x: int(x["seed"]))
    sss_by_seed: dict[int, float] = {
        int(d["seed"]): float(d["SSS"]) for d in structural_seed_details
    }
    sss_component_by_seed: dict[int, dict[str, float]] = {
        int(d["seed"]): {k: float(v) for k, v in d.get("normalized", {}).items()}
        for d in structural_seed_details
    }
    sss_breakdown: list[dict[str, Any]] = []
    for d in structural_seed_details:
        seed = int(d["seed"])
        for name, w in SSS_COMPONENT_WEIGHTS.items():
            sss_breakdown.append(
                {
                    "seed": seed,
                    "score": "SSS",
                    "component": name,
                    "weight": w,
                    "normalized_value": float(d.get("normalized", {}).get(name, 0.0)),
                    "contribution": float(d.get("contributions", {}).get(name, 0.0)),
                    "clipped": True,
                }
            )
    sss = statistics.mean(sss_by_seed.values()) if sss_by_seed else float(
        structural.get("sss", {}).get("mean", 0.0)
    )

    family_probe = variant_report.get("family_probe_summary", {})
    family_details_rows = list(family_probe.get("family_details_rows", []))
    family_summary_rows = list(family_probe.get("family_summary_rows", []))
    family_diagnostics = dict(family_probe.get("family_diagnostics", {}))
    missing_geometry_families = list(family_diagnostics.get("missing_geometry_families", []))
    geometry_family_coverage_ratio = 1.0 - (
        len(missing_geometry_families) / max(1, len(GEOMETRY_FAMILIES))
    )

    bench = variant_report.get("benchmarks", {})
    w1 = bench.get("W1_summary", {})
    w2 = bench.get("W2_summary", {})
    w3 = bench.get("W3_summary", {})
    bmap_w1 = _bench_by_seed(bench.get("W1", []))
    bmap_w2 = _bench_by_seed(bench.get("W2", []))
    bmap_w3 = _bench_by_seed(bench.get("W3", []))
    baseline_bench = baseline_report.get("benchmarks", {}) if baseline_report is not None else bench
    bbase_w1 = _bench_by_seed(baseline_bench.get("W1", []))
    bbase_w2 = _bench_by_seed(baseline_bench.get("W2", []))

    ops_by_seed: dict[int, float] = {}
    ops_breakdown: list[dict[str, Any]] = []
    candidate_seeds = sorted(set(bmap_w1.keys()).intersection(bmap_w2.keys()))
    candidate_seeds = [s for s in candidate_seeds if s in bbase_w1 and s in bbase_w2]
    for seed in candidate_seeds:
        deltas_iters: list[float] = []
        deltas_p95: list[float] = []
        deltas_rss: list[float] = []
        for wk, cur, base in (("W1", bmap_w1, bbase_w1), ("W2", bmap_w2, bbase_w2)):
            cur_m = cur[seed]
            base_m = base[seed]
            if abs(base_m["iters_per_sec"]) > 1e-12:
                deltas_iters.append(
                    ((cur_m["iters_per_sec"] - base_m["iters_per_sec"]) / base_m["iters_per_sec"]) * 100.0
                )
            if abs(base_m["p95_us"]) > 1e-12:
                deltas_p95.append(((cur_m["p95_us"] - base_m["p95_us"]) / base_m["p95_us"]) * 100.0)
            if math.isfinite(cur_m["rss_kb"]) and math.isfinite(base_m["rss_kb"]) and abs(base_m["rss_kb"]) > 1e-12:
                deltas_rss.append(((cur_m["rss_kb"] - base_m["rss_kb"]) / base_m["rss_kb"]) * 100.0)
        iters_avg_delta = statistics.mean(deltas_iters) if deltas_iters else 0.0
        p95_worst_delta = max(deltas_p95) if deltas_p95 else 0.0
        rss_worst_delta = max(deltas_rss) if deltas_rss else 0.0
        ops = clip(
            100.0
            + (0.45 * iters_avg_delta)
            - (1.25 * max(0.0, p95_worst_delta))
            - (0.65 * max(0.0, rss_worst_delta)),
            0.0,
            130.0,
        )
        ops_by_seed[seed] = ops
        ops_breakdown.extend(
            [
                {
                    "seed": seed,
                    "score": "OPS",
                    "component": "iters_avg_delta_pct",
                    "weight": 0.45,
                    "normalized_value": iters_avg_delta,
                    "contribution": 0.45 * iters_avg_delta,
                    "clipped": False,
                },
                {
                    "seed": seed,
                    "score": "OPS",
                    "component": "p95_worst_delta_pct",
                    "weight": -1.25,
                    "normalized_value": max(0.0, p95_worst_delta),
                    "contribution": -1.25 * max(0.0, p95_worst_delta),
                    "clipped": True,
                },
                {
                    "seed": seed,
                    "score": "OPS",
                    "component": "rss_worst_delta_pct",
                    "weight": -0.65,
                    "normalized_value": max(0.0, rss_worst_delta),
                    "contribution": -0.65 * max(0.0, rss_worst_delta),
                    "clipped": True,
                },
            ]
        )

    ops_mean = statistics.mean(ops_by_seed.values()) if ops_by_seed else 100.0
    bench_effects: dict[str, Any] = {}
    if baseline_report is not None:
        bbench = baseline_report.get("benchmarks", {})
        for wk in ("W1", "W2"):
            vals_base = bbench.get(f"{wk}_summary", {}).get("p95_us", {}).get("values", [])
            vals_other = bench.get(f"{wk}_summary", {}).get("p95_us", {}).get("values", [])
            if vals_base and vals_other:
                bench_effects[f"{wk}_p95_delta_pct"] = bootstrap_ci_delta_pct(
                    [float(v) for v in vals_base],
                    [float(v) for v in vals_other],
                    n_boot=bootstrap_samples,
                    seed=17 + len(wk),
                )
            vals_base = bbench.get(f"{wk}_summary", {}).get("iters_per_sec", {}).get("values", [])
            vals_other = bench.get(f"{wk}_summary", {}).get("iters_per_sec", {}).get("values", [])
            if vals_base and vals_other:
                bench_effects[f"{wk}_iters_delta_pct"] = bootstrap_ci_delta_pct(
                    [float(v) for v in vals_base],
                    [float(v) for v in vals_other],
                    n_boot=bootstrap_samples,
                    seed=29 + len(wk),
                )
            vals_base = bbench.get(f"{wk}_summary", {}).get("rss_kb", {}).get("values", [])
            vals_other = bench.get(f"{wk}_summary", {}).get("rss_kb", {}).get("values", [])
            if vals_base and vals_other:
                bench_effects[f"{wk}_rss_delta_pct"] = bootstrap_ci_delta_pct(
                    [float(v) for v in vals_base],
                    [float(v) for v in vals_other],
                    n_boot=bootstrap_samples,
                    seed=41 + len(wk),
                )

    seed_metrics_rows: list[dict[str, Any]] = []
    func_by_seed = {int(d["seed"]): d for d in functional_seed_details}
    struct_by_seed = {int(d["seed"]): d for d in structural_seed_details}
    all_seeds = sorted(set(func_by_seed.keys()) | set(struct_by_seed.keys()) | set(ops_by_seed.keys()) | set(bmap_w1.keys()) | set(bmap_w2.keys()) | set(bmap_w3.keys()))
    for seed in all_seeds:
        fd = func_by_seed.get(seed, {})
        sd = struct_by_seed.get(seed, {})
        w1s = bmap_w1.get(seed, {})
        w2s = bmap_w2.get(seed, {})
        w3s = bmap_w3.get(seed, {})
        row = {
            "seed": seed,
            "false_subthreshold_count": float(fd.get("false_subthreshold_count", float("nan"))),
            "missing_core_rate": float(fd.get("missing_core_rate", float("nan"))),
            "missing_cau_count": float(fd.get("missing_cau_count", float("nan"))),
            "missing_ctf_count": float(fd.get("missing_ctf_count", float("nan"))),
            "multiscale_convergent_ratio": float(sd.get("raw", {}).get("convergent_ratio", float("nan"))),
            "multiscale_pathological_count": float(sd.get("raw", {}).get("pathological_multiscale_count", float("nan"))),
            "box_pathological_count": float(sd.get("raw", {}).get("pathological_box_counting_count", float("nan"))),
            "temporal_self_similar_ratio": float(sd.get("raw", {}).get("self_similar_ratio", float("nan"))),
            "temporal_scale_error": float(sd.get("raw", {}).get("temporal_scale_error", float("nan"))),
            "avalanche_dominant_fraction": float(sd.get("raw", {}).get("dominant_fraction", float("nan"))),
            "W1_iters_per_sec": float(w1s.get("iters_per_sec", float("nan"))),
            "W1_p95_us": float(w1s.get("p95_us", float("nan"))),
            "W1_rss_kb": float(w1s.get("rss_kb", float("nan"))),
            "W2_iters_per_sec": float(w2s.get("iters_per_sec", float("nan"))),
            "W2_p95_us": float(w2s.get("p95_us", float("nan"))),
            "W2_rss_kb": float(w2s.get("rss_kb", float("nan"))),
            "W3_iters_per_sec": float(w3s.get("iters_per_sec", float("nan"))),
            "W3_p95_us": float(w3s.get("p95_us", float("nan"))),
            "W3_rss_kb": float(w3s.get("rss_kb", float("nan"))),
            "FHS": float(fhs_by_seed.get(seed, float("nan"))),
            "SSS": float(sss_by_seed.get(seed, float("nan"))),
            "OPS": float(ops_by_seed.get(seed, float("nan"))),
        }
        seed_metrics_rows.append(row)

    return {
        "common_total": common_total,
        "common_passed": common_passed,
        "common_failed": common_failed,
        "common_errors": common_errors,
        "common_pass_rate": common_rate,
        "common_pass_rate_wilson": common_wilson,
        "ordered_total": ordered_total,
        "ordered_passed": ordered_passed,
        "ordered_pass_rate": ordered_rate,
        "ordered_pass_rate_wilson": ordered_wilson,
        "false_subthreshold_mean": false_sub_mean,
        "missing_core_rate_mean": missing_core_rate,
        "functional_regression": functional_regression,
        "FHS": fhs,
        "FHS_runs": [fhs_by_seed[s] for s in sorted(fhs_by_seed)],
        "FHS_by_seed": fhs_by_seed,
        "SSS": sss,
        "SSS_runs": [sss_by_seed[s] for s in sorted(sss_by_seed)],
        "SSS_by_seed": sss_by_seed,
        "SSS_components_by_seed": sss_component_by_seed,
        "OPS": ops_mean,
        "OPS_runs": [ops_by_seed[s] for s in sorted(ops_by_seed)],
        "OPS_by_seed": ops_by_seed,
        "W1_p95_mean": float(w1.get("p95_us", {}).get("mean", float("nan"))),
        "W2_p95_mean": float(w2.get("p95_us", {}).get("mean", float("nan"))),
        "W3_p95_mean": float(w3.get("p95_us", {}).get("mean", float("nan"))),
        "W1_iters_mean": float(w1.get("iters_per_sec", {}).get("mean", float("nan"))),
        "W2_iters_mean": float(w2.get("iters_per_sec", {}).get("mean", float("nan"))),
        "W3_iters_mean": float(w3.get("iters_per_sec", {}).get("mean", float("nan"))),
        "W1_rss_mean": float(w1.get("rss_kb", {}).get("mean", float("nan"))),
        "W2_rss_mean": float(w2.get("rss_kb", {}).get("mean", float("nan"))),
        "W3_rss_mean": float(w3.get("rss_kb", {}).get("mean", float("nan"))),
        "pathological_multiscale_mean": pathological_multiscale,
        "temporal_scale_error_mean": temporal_scale_error,
        "avalanche_dominant_fraction_mean": dominant_fraction,
        "geometry_family_coverage_ratio": geometry_family_coverage_ratio,
        "missing_geometry_families": missing_geometry_families,
        "bench_effects_vs_a1": bench_effects,
        "seed_metrics_rows": seed_metrics_rows,
        "score_breakdown_rows": fhs_breakdown + sss_breakdown + ops_breakdown,
        "family_details_rows": family_details_rows,
        "family_summary_rows": family_summary_rows,
        "family_diagnostics": family_diagnostics,
    }


def effect_bundle(
    *,
    base: dict[str, Any],
    other: dict[str, Any],
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    min_n: int = 3,
) -> dict[str, Any]:
    fhs = _metric_effect_from_seed_maps(
        base_map={int(k): float(v) for k, v in base.get("FHS_by_seed", {}).items()},
        other_map={int(k): float(v) for k, v in other.get("FHS_by_seed", {}).items()},
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=seed,
        min_n=min_n,
    )
    sss = _metric_effect_from_seed_maps(
        base_map={int(k): float(v) for k, v in base.get("SSS_by_seed", {}).items()},
        other_map={int(k): float(v) for k, v in other.get("SSS_by_seed", {}).items()},
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=seed + 17,
        min_n=min_n,
    )
    ops = _metric_effect_from_seed_maps(
        base_map={int(k): float(v) for k, v in base.get("OPS_by_seed", {}).items()},
        other_map={int(k): float(v) for k, v in other.get("OPS_by_seed", {}).items()},
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=seed + 29,
        min_n=min_n,
    )
    return {
        "FHS_delta": fhs["delta"],
        "SSS_delta": sss["delta"],
        "OPS_delta": ops["delta"],
        "SSS_hedges_g": sss["hedges_g"],
        "SSS_cliffs_delta": sss["cliffs_delta"],
        "SSS_permutation": sss["permutation"],
        "diagnostics": {
            "FHS": fhs["diagnostics"],
            "SSS": sss["diagnostics"],
            "OPS": ops["diagnostics"],
        },
    }


def run_sensitivity_analysis(
    *,
    a1: dict[str, Any],
    a5: dict[str, Any],
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict[str, Any]:
    base_components = {int(k): v for k, v in a1.get("SSS_components_by_seed", {}).items()}
    other_components = {int(k): v for k, v in a5.get("SSS_components_by_seed", {}).items()}
    shared = sorted(set(base_components.keys()).intersection(other_components.keys()))
    if len(shared) < 3:
        return {"status": "insufficient_n", "shared_seeds": shared, "summary": {"stable_positive_fraction": 0.0, "has_large_negative": True}}

    base_sss = {s: float(a1["SSS_by_seed"][s]) for s in shared if s in a1.get("SSS_by_seed", {})}
    other_sss = {s: float(a5["SSS_by_seed"][s]) for s in shared if s in a5.get("SSS_by_seed", {})}

    leave_seed_out: list[dict[str, Any]] = []
    for seed in shared:
        b = {k: v for k, v in base_sss.items() if k != seed}
        o = {k: v for k, v in other_sss.items() if k != seed}
        eff = _metric_effect_from_seed_maps(
            base_map=b,
            other_map=o,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            seed=700 + seed,
            min_n=2,
        )
        leave_seed_out.append(
            {
                "left_out_seed": seed,
                "delta_point": eff["delta"]["point"],
                "ci95_low": eff["delta"]["ci95_low"],
                "ci95_high": eff["delta"]["ci95_high"],
                "p_perm": eff["permutation"]["p_two_sided"],
            }
        )

    leave_probe_out: list[dict[str, Any]] = []
    for component in SSS_COMPONENT_WEIGHTS:
        b_map: dict[int, float] = {}
        o_map: dict[int, float] = {}
        for seed in shared:
            b_score, _ = compute_sss_from_components(base_components[seed], excluded={component})
            o_score, _ = compute_sss_from_components(other_components[seed], excluded={component})
            b_map[seed] = b_score
            o_map[seed] = o_score
        eff = _metric_effect_from_seed_maps(
            base_map=b_map,
            other_map=o_map,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            seed=1700 + len(component),
            min_n=3,
        )
        leave_probe_out.append(
            {
                "excluded_probe": component,
                "delta_point": eff["delta"]["point"],
                "ci95_low": eff["delta"]["ci95_low"],
                "ci95_high": eff["delta"]["ci95_high"],
                "p_perm": eff["permutation"]["p_two_sided"],
            }
        )

    leave_family_out: list[dict[str, Any]] = []
    families = sorted(set(SSS_COMPONENT_FAMILIES.values()))
    for fam in families:
        excluded = {k for k, v in SSS_COMPONENT_FAMILIES.items() if v == fam}
        b_map: dict[int, float] = {}
        o_map: dict[int, float] = {}
        for seed in shared:
            b_score, _ = compute_sss_from_components(base_components[seed], excluded=excluded)
            o_score, _ = compute_sss_from_components(other_components[seed], excluded=excluded)
            b_map[seed] = b_score
            o_map[seed] = o_score
        eff = _metric_effect_from_seed_maps(
            base_map=b_map,
            other_map=o_map,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            seed=2700 + len(fam),
            min_n=3,
        )
        leave_family_out.append(
            {
                "excluded_family": fam,
                "delta_point": eff["delta"]["point"],
                "ci95_low": eff["delta"]["ci95_low"],
                "ci95_high": eff["delta"]["ci95_high"],
                "p_perm": eff["permutation"]["p_two_sided"],
            }
        )

    all_cases = leave_seed_out + leave_probe_out + leave_family_out
    pos_ci = [c for c in all_cases if float(c["ci95_low"]) > 0.0]
    has_large_negative = any(float(c["delta_point"]) < -1.0 for c in all_cases)
    return {
        "status": "ok",
        "shared_seeds": shared,
        "leave_one_seed_out": leave_seed_out,
        "leave_one_probe_out": leave_probe_out,
        "leave_one_metric_family_out": leave_family_out,
        "summary": {
            "n_cases": len(all_cases),
            "stable_positive_fraction": (len(pos_ci) / len(all_cases)) if all_cases else 0.0,
            "has_large_negative": has_large_negative,
        },
    }


def run_negative_controls(
    *,
    a1: dict[str, Any],
    a5: dict[str, Any],
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict[str, Any]:
    base_components = {int(k): v for k, v in a1.get("SSS_components_by_seed", {}).items()}
    other_components = {int(k): v for k, v in a5.get("SSS_components_by_seed", {}).items()}
    shared = sorted(set(base_components.keys()).intersection(other_components.keys()))
    if len(shared) < 3:
        return {"status": "insufficient_n", "shared_seeds": shared, "replicates_effect": False}

    base_sss_map = {s: float(a1["SSS_by_seed"][s]) for s in shared if s in a1.get("SSS_by_seed", {})}
    other_sss_map = {s: float(a5["SSS_by_seed"][s]) for s in shared if s in a5.get("SSS_by_seed", {})}
    component_values: dict[str, list[float]] = {name: [] for name in SSS_COMPONENT_WEIGHTS}
    for seed in shared:
        for name in SSS_COMPONENT_WEIGHTS:
            component_values[name].append(float(other_components[seed][name]))

    rng = random.Random(556677)
    sham_map: dict[int, float] = {}
    for seed in shared:
        comp: dict[str, float] = {}
        for name, vals in component_values.items():
            comp[name] = vals[rng.randrange(len(vals))]
        score, _ = compute_sss_from_components(comp)
        sham_map[seed] = score

    randomized_map: dict[int, float] = {s: 0.0 for s in shared}
    for name, vals in component_values.items():
        perm = vals[:]
        rng.shuffle(perm)
        for i, seed in enumerate(shared):
            randomized_map[seed] += 100.0 * SSS_COMPONENT_WEIGHTS[name] * clip(float(perm[i]), 0.0, 1.0)
    for seed in shared:
        randomized_map[seed] = clip(randomized_map[seed], 0.0, 100.0)

    real_eff = _metric_effect_from_seed_maps(
        base_map=base_sss_map,
        other_map=other_sss_map,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=901,
        min_n=3,
    )
    sham_eff = _metric_effect_from_seed_maps(
        base_map=base_sss_map,
        other_map=sham_map,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=902,
        min_n=3,
    )
    rand_eff = _metric_effect_from_seed_maps(
        base_map=base_sss_map,
        other_map=randomized_map,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=903,
        min_n=3,
    )

    real_delta = float(real_eff["delta"]["point"])
    sham_delta = float(sham_eff["delta"]["point"])
    rand_delta = float(rand_eff["delta"]["point"])
    sham_sig = float(sham_eff["permutation"]["p_two_sided"]) < 0.05 and float(sham_eff["delta"]["ci95_low"]) > 0.0
    rand_sig = float(rand_eff["permutation"]["p_two_sided"]) < 0.05 and float(rand_eff["delta"]["ci95_low"]) > 0.0
    replicate_scale = max(1e-9, abs(real_delta))
    replicates_effect = (
        (sham_sig and abs(sham_delta) >= 0.8 * replicate_scale)
        or (rand_sig and abs(rand_delta) >= 0.8 * replicate_scale)
    )
    return {
        "status": "ok",
        "shared_seeds": shared,
        "real_effect": {
            "delta": real_eff["delta"],
            "permutation": real_eff["permutation"],
        },
        "sham_fractal": {
            "delta": sham_eff["delta"],
            "permutation": sham_eff["permutation"],
        },
        "randomized_fractal": {
            "delta": rand_eff["delta"],
            "permutation": rand_eff["permutation"],
        },
        "replicates_effect": replicates_effect,
    }


def classify_outcome(
    *,
    rows_by_id: dict[str, dict[str, Any]],
    effects: dict[str, dict[str, Any]],
    have_all: bool,
    sensitivity: dict[str, Any],
    negative_controls: dict[str, Any],
) -> tuple[str, str]:
    if not have_all:
        return "No concluyente", "No se ejecutaron A1..A5 completos; evidencia insuficiente para inferencia causal."

    a1 = rows_by_id["A1_baseline_puro"]
    a5 = rows_by_id["A5_fractal_full"]
    if a5["functional_regression"]:
        return "Regresión", "Gate funcional violado en régimen común."

    net = effects["A5_vs_A1"]
    sss_delta = net["SSS_delta"]
    sss_diag = net["diagnostics"]["SSS"]
    sss_perm = net["SSS_permutation"]
    sss_g = net["SSS_hedges_g"]
    sss_cliff = net["SSS_cliffs_delta"]

    ops_comparable = (
        a5["OPS"] >= (a1["OPS"] - 5.0)
        and a5["common_pass_rate"] >= a1["common_pass_rate"]
        and a5["false_subthreshold_mean"] <= a1["false_subthreshold_mean"]
    )
    fhs_ok = a5["FHS"] >= a1["FHS"]

    robust_stats = (
        bool(sss_diag.get("enough_n"))
        and not bool(sss_diag.get("ci_degenerate"))
        and not bool(sss_diag.get("delta_has_nan"))
        and float(sss_delta.get("ci95_low", float("nan"))) > 0.0
        and math.isfinite(float(sss_perm.get("p_two_sided", float("nan"))))
        and float(sss_perm.get("p_two_sided")) < 0.05
        and not bool(sss_g.get("undefined", True))
        and math.isfinite(float(sss_g.get("value", float("nan"))))
        and float(sss_g.get("value")) > 0.0
        and math.isfinite(float(sss_cliff.get("value", float("nan"))))
        and float(sss_cliff.get("value")) > 0.0
        and bool(sss_diag.get("sign_consistent"))
    )
    sens_summary = sensitivity.get("summary", {})
    sensitivity_ok = (
        sensitivity.get("status") == "ok"
        and float(sens_summary.get("stable_positive_fraction", 0.0)) >= 0.8
        and not bool(sens_summary.get("has_large_negative", True))
    )
    controls_ok = (
        negative_controls.get("status") == "ok"
        and not bool(negative_controls.get("replicates_effect", True))
    )

    if robust_stats and sensitivity_ok and controls_ok and fhs_ok and ops_comparable:
        return (
            "Superioridad fractal",
            "A5 supera A1 en SSS con inferencia coherente (CI95 no degenerado, p_perm<0.05), sin regresión funcional y sin replicación en controles negativos.",
        )

    if fhs_ok and ops_comparable and float(sss_delta.get("point", 0.0)) > 0.0:
        if robust_stats and (not controls_ok or not sensitivity_ok):
            return (
                "Comparabilidad con firma estructural distinta",
                "Existe señal estructural favorable, pero no pasó completamente robustez de sensibilidad/controles negativos.",
            )
        return (
            "No concluyente",
            "Hay señal estructural, pero la robustez estadística/causal no es suficiente para superioridad.",
        )

    if not ops_comparable:
        return "Regresión", "No se mantuvo comparabilidad operacional frente a A1."

    return "No concluyente", "Diferencias insuficientes o inconsistentes para concluir aporte fractal robusto."


def export_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "variant_id",
        "label",
        "bridge_mode",
        "policy_ref",
        "bridge_ref",
        "metrics_ref",
        "common_pass_rate",
        "common_pass_rate_wilson_low",
        "common_pass_rate_wilson_high",
        "ordered_pass_rate",
        "ordered_pass_rate_wilson_low",
        "ordered_pass_rate_wilson_high",
        "false_subthreshold_mean",
        "missing_core_rate_mean",
        "FHS",
        "SSS",
        "OPS",
        "W1_iters_mean",
        "W1_p95_mean",
        "W1_rss_mean",
        "W2_iters_mean",
        "W2_p95_mean",
        "W2_rss_mean",
        "W3_iters_mean",
        "W3_p95_mean",
        "W3_rss_mean",
        "pathological_multiscale_mean",
        "temporal_scale_error_mean",
        "avalanche_dominant_fraction_mean",
        "functional_regression",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant_id": row["variant_id"],
                    "label": row["label"],
                    "bridge_mode": row["bridge_mode"],
                    "policy_ref": row["policy_ref"],
                    "bridge_ref": row["bridge_ref"],
                    "metrics_ref": row["metrics_ref"],
                    "common_pass_rate": row["common_pass_rate"],
                    "common_pass_rate_wilson_low": row["common_pass_rate_wilson"]["low"],
                    "common_pass_rate_wilson_high": row["common_pass_rate_wilson"]["high"],
                    "ordered_pass_rate": row["ordered_pass_rate"],
                    "ordered_pass_rate_wilson_low": row["ordered_pass_rate_wilson"]["low"],
                    "ordered_pass_rate_wilson_high": row["ordered_pass_rate_wilson"]["high"],
                    "false_subthreshold_mean": row["false_subthreshold_mean"],
                    "missing_core_rate_mean": row["missing_core_rate_mean"],
                    "FHS": row["FHS"],
                    "SSS": row["SSS"],
                    "OPS": row["OPS"],
                    "W1_iters_mean": row["W1_iters_mean"],
                    "W1_p95_mean": row["W1_p95_mean"],
                    "W1_rss_mean": row["W1_rss_mean"],
                    "W2_iters_mean": row["W2_iters_mean"],
                    "W2_p95_mean": row["W2_p95_mean"],
                    "W2_rss_mean": row["W2_rss_mean"],
                    "W3_iters_mean": row["W3_iters_mean"],
                    "W3_p95_mean": row["W3_p95_mean"],
                    "W3_rss_mean": row["W3_rss_mean"],
                    "pathological_multiscale_mean": row["pathological_multiscale_mean"],
                    "temporal_scale_error_mean": row["temporal_scale_error_mean"],
                    "avalanche_dominant_fraction_mean": row["avalanche_dominant_fraction_mean"],
                    "functional_regression": row["functional_regression"],
                }
            )


def export_per_seed_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "variant_id",
        "seed",
        "false_subthreshold_count",
        "missing_core_rate",
        "missing_cau_count",
        "missing_ctf_count",
        "multiscale_convergent_ratio",
        "multiscale_pathological_count",
        "box_pathological_count",
        "temporal_self_similar_ratio",
        "temporal_scale_error",
        "avalanche_dominant_fraction",
        "W1_iters_per_sec",
        "W1_p95_us",
        "W1_rss_kb",
        "W2_iters_per_sec",
        "W2_p95_us",
        "W2_rss_kb",
        "W3_iters_per_sec",
        "W3_p95_us",
        "W3_rss_kb",
        "FHS",
        "SSS",
        "OPS",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for variant in rows:
            for item in variant.get("seed_metrics_rows", []):
                writer.writerow({"variant_id": variant["variant_id"], **item})


def export_score_breakdown_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "variant_id",
        "seed",
        "score",
        "component",
        "weight",
        "normalized_value",
        "contribution",
        "clipped",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for variant in rows:
            for item in variant.get("score_breakdown_rows", []):
                writer.writerow({"variant_id": variant["variant_id"], **item})


def export_family_details_long_csv(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = [
        "variant_id",
        "seed",
        "family_type",
        "family_id",
        "geometry_name",
        "geometry_family",
        "metric",
        "value",
        "applicable",
        "present",
        "position",
    ]
    out: list[dict[str, Any]] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for variant in rows:
            for item in variant.get("family_details_rows", []):
                row = {"variant_id": variant["variant_id"], **item}
                out.append(row)
                writer.writerow(row)
    return out


def export_family_summary_csv(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = [
        "variant_id",
        "seed",
        "family_type",
        "family_id",
        "metric",
        "mean",
        "stdev",
        "n",
    ]
    out: list[dict[str, Any]] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for variant in rows:
            for item in variant.get("family_summary_rows", []):
                row = {"variant_id": variant["variant_id"], **item}
                out.append(row)
                writer.writerow(row)
    return out


def collect_family_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in rows:
        diag = variant.get("family_diagnostics", {})
        by_variant[variant["variant_id"]] = diag
    a1 = by_variant.get("A1_baseline_puro", {})
    a4 = by_variant.get("A4_metrics_legacy", {})
    a5 = by_variant.get("A5_fractal_full", {})
    return {
        "status": "ok",
        "variants": by_variant,
        "focus": {
            "A1_missing_geometry_families": a1.get("missing_geometry_families", []),
            "A4_missing_geometry_families": a4.get("missing_geometry_families", []),
            "A5_missing_geometry_families": a5.get("missing_geometry_families", []),
        },
    }


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _chart_record(*, out_dir: Path, chart_id: str, path: Path, series: dict[str, Any]) -> dict[str, Any]:
    return {
        "chart_id": chart_id,
        "relative_path": str(path.relative_to(out_dir)),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "created_at": datetime.now().isoformat(),
        "series": series,
    }


def _draw_grouped_bar_png(
    *,
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1280, 720
    margin_left, margin_right = 100, 40
    margin_top, margin_bottom = 80, 140
    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    all_vals = [0.0]
    for _, vals, _ in series:
        for v in vals:
            if math.isfinite(v):
                all_vals.append(v)
    v_min = min(all_vals) if y_min is None else y_min
    v_max = max(all_vals) if y_max is None else y_max
    if not math.isfinite(v_min):
        v_min = 0.0
    if not math.isfinite(v_max):
        v_max = 1.0
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1.0
    if v_min > 0.0:
        v_min = 0.0
    if v_max < 0.0:
        v_max = 0.0

    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    x0, y0 = margin_left, margin_top
    x1, y1 = margin_left + chart_w, margin_top + chart_h

    draw.rectangle((x0, y0, x1, y1), outline=(120, 120, 120), width=1)
    draw.text((margin_left, 20), title, fill=(20, 20, 20), font=font)

    def y_px(v: float) -> float:
        return y0 + (v_max - v) * (chart_h / (v_max - v_min))

    zero_y = y_px(0.0)
    draw.line((x0, zero_y, x1, zero_y), fill=(80, 80, 80), width=2)

    n_labels = max(1, len(labels))
    n_series = max(1, len(series))
    group_w = chart_w / n_labels
    bar_w = max(4.0, group_w / (n_series + 1.5))

    for i, label in enumerate(labels):
        gx = x0 + i * group_w + group_w * 0.15
        for j, (_, vals, color) in enumerate(series):
            if i >= len(vals):
                continue
            val = vals[i]
            if not math.isfinite(val):
                continue
            bx0 = gx + j * bar_w
            bx1 = bx0 + bar_w * 0.9
            by = y_px(val)
            top = min(by, zero_y)
            bot = max(by, zero_y)
            draw.rectangle((bx0, top, bx1, bot), fill=color, outline=(90, 90, 90))
        draw.text((x0 + i * group_w + 2, y1 + 8), label[:18], fill=(30, 30, 30), font=font)

    leg_x = width - margin_right - 220
    leg_y = margin_top + 8
    for idx, (name, _, color) in enumerate(series):
        ly = leg_y + idx * 18
        draw.rectangle((leg_x, ly, leg_x + 12, ly + 12), fill=color, outline=(80, 80, 80))
        draw.text((leg_x + 18, ly), name, fill=(30, 30, 30), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def _draw_heatmap_png(
    *,
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[float]],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1100, 760
    margin_left, margin_right = 140, 60
    margin_top, margin_bottom = 90, 90
    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((margin_left, 24), title, fill=(20, 20, 20), font=font)

    rows = max(1, len(row_labels))
    cols = max(1, len(col_labels))
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    cell_w = chart_w / cols
    cell_h = chart_h / rows

    flat = [v for line in matrix for v in line if isinstance(v, (int, float)) and math.isfinite(float(v))]
    v_min = min(flat) if flat else 0.0
    v_max = max(flat) if flat else 1.0
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1.0

    def color_for(v: float) -> tuple[int, int, int]:
        if not math.isfinite(v):
            return (220, 220, 220)
        t = (v - v_min) / (v_max - v_min)
        t = max(0.0, min(1.0, t))
        r = int(50 + 180 * t)
        g = int(80 + 120 * (1.0 - abs(t - 0.5) * 2.0))
        b = int(180 - 140 * t)
        return (r, g, b)

    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            x0 = margin_left + j * cell_w
            y0 = margin_top + i * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            draw.rectangle((x0, y0, x1, y1), fill=color_for(float(v)), outline=(180, 180, 180))
            draw.text((x0 + 6, y0 + 4), f"{float(v):.2f}" if math.isfinite(float(v)) else "nan", fill=(20, 20, 20), font=font)

    for i, label in enumerate(row_labels):
        y = margin_top + i * cell_h + cell_h * 0.35
        draw.text((12, y), label, fill=(30, 30, 30), font=font)
    for j, label in enumerate(col_labels):
        x = margin_left + j * cell_w + 6
        draw.text((x, margin_top - 22), label, fill=(30, 30, 30), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def _generate_comparative_charts_pillow(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    effects: dict[str, dict[str, Any]],
    family_details_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []

    variant_ids = [r["variant_id"] for r in rows]
    fhs_vals = [_safe_float(r.get("FHS")) for r in rows]
    sss_vals = [_safe_float(r.get("SSS")) for r in rows]
    ops_vals = [_safe_float(r.get("OPS")) for r in rows]
    path = charts_dir / "scores_by_variant.png"
    _draw_grouped_bar_png(
        path=path,
        title="Scores por variante",
        labels=variant_ids,
        series=[
            ("FHS", fhs_vals, (88, 140, 210)),
            ("SSS", sss_vals, (90, 190, 130)),
            ("OPS", ops_vals, (240, 170, 80)),
        ],
    )
    manifests.append(_chart_record(out_dir=out_dir, chart_id="scores_by_variant", path=path, series={"variant_ids": variant_ids, "FHS": fhs_vals, "SSS": sss_vals, "OPS": ops_vals}))

    focus = {"A1_baseline_puro", "A4_metrics_legacy", "A5_fractal_full"}
    comp_data: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        vid = row["variant_id"]
        if vid not in focus:
            continue
        for item in row.get("score_breakdown_rows", []):
            if item.get("score") != "OPS":
                continue
            comp = str(item.get("component"))
            comp_data.setdefault(comp, {}).setdefault(vid, []).append(_safe_float(item.get("contribution")))
    comps = sorted(comp_data.keys())
    labels = comps
    a1_vals = [statistics.mean([v for v in comp_data.get(c, {}).get("A1_baseline_puro", []) if math.isfinite(v)]) if comp_data.get(c, {}).get("A1_baseline_puro") else 0.0 for c in comps]
    a4_vals = [statistics.mean([v for v in comp_data.get(c, {}).get("A4_metrics_legacy", []) if math.isfinite(v)]) if comp_data.get(c, {}).get("A4_metrics_legacy") else 0.0 for c in comps]
    a5_vals = [statistics.mean([v for v in comp_data.get(c, {}).get("A5_fractal_full", []) if math.isfinite(v)]) if comp_data.get(c, {}).get("A5_fractal_full") else 0.0 for c in comps]
    path = charts_dir / "ops_components_A1_A4_A5.png"
    _draw_grouped_bar_png(
        path=path,
        title="OPS por componente (A1 vs A4 vs A5)",
        labels=labels,
        series=[
            ("A1", a1_vals, (98, 114, 164)),
            ("A4", a4_vals, (80, 220, 120)),
            ("A5", a5_vals, (250, 180, 100)),
        ],
    )
    manifests.append(_chart_record(out_dir=out_dir, chart_id="ops_components_A1_A4_A5", path=path, series={"components": labels, "A1": a1_vals, "A4": a4_vals, "A5": a5_vals}))

    geom_focus = ["A1_baseline_puro", "A4_metrics_legacy", "A5_fractal_full"]
    geom_map: dict[str, dict[str, list[float]]] = {v: {f: [] for f in GEOMETRY_FAMILIES} for v in geom_focus}
    for item in family_details_rows:
        if item.get("family_type") != "geometry" or item.get("metric") != "atlas_overall":
            continue
        vid = str(item.get("variant_id"))
        fam = str(item.get("family_id"))
        val = _safe_float(item.get("value"))
        if vid in geom_map and fam in geom_map[vid] and math.isfinite(val):
            geom_map[vid][fam].append(val)
    matrix: list[list[float]] = []
    for fam in GEOMETRY_FAMILIES:
        row_vals = []
        for vid in geom_focus:
            xs = geom_map.get(vid, {}).get(fam, [])
            row_vals.append(statistics.mean(xs) if xs else float("nan"))
        matrix.append(row_vals)
    path = charts_dir / "geometry_family_heatmap_A1_A4_A5.png"
    _draw_heatmap_png(
        path=path,
        title="Señal estructural por familia geométrica (atlas_overall)",
        row_labels=GEOMETRY_FAMILIES,
        col_labels=["A1", "A4", "A5"],
        matrix=matrix,
    )
    manifests.append(_chart_record(out_dir=out_dir, chart_id="geometry_family_heatmap_A1_A4_A5", path=path, series={"matrix": matrix, "families": GEOMETRY_FAMILIES, "variants": geom_focus}))

    reas_map: dict[str, dict[str, list[float]]] = {v: {f: [] for f in REASONING_FAMILIES} for v in geom_focus}
    for item in family_details_rows:
        if item.get("family_type") != "reasoning" or item.get("metric") != "present":
            continue
        if not bool(item.get("applicable", False)):
            continue
        vid = str(item.get("variant_id"))
        fam = str(item.get("family_id"))
        val = _safe_float(item.get("value"))
        if vid in reas_map and fam in reas_map[vid] and math.isfinite(val):
            reas_map[vid][fam].append(val)
    a1 = [statistics.mean(reas_map["A1_baseline_puro"][f]) if reas_map["A1_baseline_puro"][f] else 0.0 for f in REASONING_FAMILIES]
    a4 = [statistics.mean(reas_map["A4_metrics_legacy"][f]) if reas_map["A4_metrics_legacy"][f] else 0.0 for f in REASONING_FAMILIES]
    a5 = [statistics.mean(reas_map["A5_fractal_full"][f]) if reas_map["A5_fractal_full"][f] else 0.0 for f in REASONING_FAMILIES]
    path = charts_dir / "reasoning_family_activation_A1_A4_A5.png"
    _draw_grouped_bar_png(
        path=path,
        title="Activación por familia de reasoning (A1/A4/A5)",
        labels=REASONING_FAMILIES,
        series=[
            ("A1", a1, (98, 114, 164)),
            ("A4", a4, (80, 220, 120)),
            ("A5", a5, (250, 180, 100)),
        ],
        y_min=0.0,
        y_max=1.0,
    )
    manifests.append(_chart_record(out_dir=out_dir, chart_id="reasoning_family_activation_A1_A4_A5", path=path, series={"A1": a1, "A4": a4, "A5": a5, "families": REASONING_FAMILIES}))

    labels = ["A2-A1", "A5-A4", "A5-A1"]
    keys = ["A2_vs_A1", "A5_vs_A4", "A5_vs_A1"]
    fhs = [_safe_float(effects.get(k, {}).get("FHS_delta", {}).get("point")) for k in keys]
    sss = [_safe_float(effects.get(k, {}).get("SSS_delta", {}).get("point")) for k in keys]
    ops = [_safe_float(effects.get(k, {}).get("OPS_delta", {}).get("point")) for k in keys]
    path = charts_dir / "bridge_effect_vs_metrics_effect.png"
    _draw_grouped_bar_png(
        path=path,
        title="Comparación causal: bridge vs métricas",
        labels=labels,
        series=[
            ("ΔFHS", fhs, (88, 140, 210)),
            ("ΔSSS", sss, (90, 190, 130)),
            ("ΔOPS", ops, (240, 120, 90)),
        ],
    )
    manifests.append(_chart_record(out_dir=out_dir, chart_id="bridge_effect_vs_metrics_effect", path=path, series={"labels": labels, "delta_FHS": fhs, "delta_SSS": sss, "delta_OPS": ops}))

    return manifests


def generate_comparative_charts(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    effects: dict[str, dict[str, Any]],
    family_details_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _generate_comparative_charts_pillow(
            out_dir=out_dir,
            rows=rows,
            effects=effects,
            family_details_rows=family_details_rows,
        )

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []

    # 1) Scores por variante (FHS/SSS/OPS)
    variant_ids = [r["variant_id"] for r in rows]
    fhs_vals = [_safe_float(r.get("FHS")) for r in rows]
    sss_vals = [_safe_float(r.get("SSS")) for r in rows]
    ops_vals = [_safe_float(r.get("OPS")) for r in rows]
    x = list(range(len(rows)))
    w = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - w for i in x], fhs_vals, width=w, label="FHS")
    ax.bar(x, sss_vals, width=w, label="SSS")
    ax.bar([i + w for i in x], ops_vals, width=w, label="OPS")
    ax.set_xticks(x)
    ax.set_xticklabels(variant_ids, rotation=20, ha="right")
    ax.set_ylim(0, max(130.0, max([v for v in fhs_vals + sss_vals + ops_vals if math.isfinite(v)] + [100.0]) * 1.1))
    ax.set_title("Scores por variante")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    path = charts_dir / "scores_by_variant.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    manifests.append(
        _chart_record(
            out_dir=out_dir,
            chart_id="scores_by_variant",
            path=path,
            series={"variant_ids": variant_ids, "FHS": fhs_vals, "SSS": sss_vals, "OPS": ops_vals},
        )
    )

    # 2) Componentes OPS A1/A4/A5
    focus = {"A1_baseline_puro", "A4_metrics_legacy", "A5_fractal_full"}
    comp_data: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        vid = row["variant_id"]
        if vid not in focus:
            continue
        for item in row.get("score_breakdown_rows", []):
            if item.get("score") != "OPS":
                continue
            comp = str(item.get("component"))
            comp_data.setdefault(comp, {}).setdefault(vid, []).append(_safe_float(item.get("contribution")))
    comps = sorted(comp_data.keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    bar_x = list(range(len(comps)))
    offs = {"A1_baseline_puro": -0.22, "A4_metrics_legacy": 0.0, "A5_fractal_full": 0.22}
    labels = {
        "A1_baseline_puro": "A1",
        "A4_metrics_legacy": "A4",
        "A5_fractal_full": "A5",
    }
    for vid in ("A1_baseline_puro", "A4_metrics_legacy", "A5_fractal_full"):
        vals = []
        for comp in comps:
            xs = [v for v in comp_data.get(comp, {}).get(vid, []) if math.isfinite(v)]
            vals.append(statistics.mean(xs) if xs else 0.0)
        ax.bar([i + offs[vid] for i in bar_x], vals, width=0.2, label=labels[vid])
    ax.set_xticks(bar_x)
    ax.set_xticklabels(comps, rotation=18, ha="right")
    ax.set_title("OPS por componente (A1 vs A4 vs A5)")
    ax.grid(alpha=0.2, axis="y")
    ax.legend()
    path = charts_dir / "ops_components_A1_A4_A5.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    manifests.append(
        _chart_record(
            out_dir=out_dir,
            chart_id="ops_components_A1_A4_A5",
            path=path,
            series=comp_data,
        )
    )

    # 3) Heatmap geometría por familia (atlas_overall)
    geom_focus = ["A1_baseline_puro", "A4_metrics_legacy", "A5_fractal_full"]
    geom_map: dict[str, dict[str, list[float]]] = {v: {f: [] for f in GEOMETRY_FAMILIES} for v in geom_focus}
    for item in family_details_rows:
        if item.get("family_type") != "geometry" or item.get("metric") != "atlas_overall":
            continue
        vid = str(item.get("variant_id"))
        fam = str(item.get("family_id"))
        val = _safe_float(item.get("value"))
        if vid in geom_map and fam in geom_map[vid] and math.isfinite(val):
            geom_map[vid][fam].append(val)
    matrix: list[list[float]] = []
    for fam in GEOMETRY_FAMILIES:
        row_vals = []
        for vid in geom_focus:
            xs = geom_map.get(vid, {}).get(fam, [])
            row_vals.append(statistics.mean(xs) if xs else float("nan"))
        matrix.append(row_vals)
    import numpy as np

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.nan_to_num(arr, nan=0.0), cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(GEOMETRY_FAMILIES)))
    ax.set_yticklabels(GEOMETRY_FAMILIES)
    ax.set_xticks(range(len(geom_focus)))
    ax.set_xticklabels(["A1", "A4", "A5"])
    ax.set_title("Señal estructural por familia geométrica (atlas_overall)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    path = charts_dir / "geometry_family_heatmap_A1_A4_A5.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    manifests.append(
        _chart_record(
            out_dir=out_dir,
            chart_id="geometry_family_heatmap_A1_A4_A5",
            path=path,
            series={"matrix": matrix, "families": GEOMETRY_FAMILIES, "variants": geom_focus},
        )
    )

    # 4) Activación reasoning por familia (present metric)
    reas_map: dict[str, dict[str, list[float]]] = {v: {f: [] for f in REASONING_FAMILIES} for v in geom_focus}
    for item in family_details_rows:
        if item.get("family_type") != "reasoning" or item.get("metric") != "present":
            continue
        if not bool(item.get("applicable", False)):
            continue
        vid = str(item.get("variant_id"))
        fam = str(item.get("family_id"))
        val = _safe_float(item.get("value"))
        if vid in reas_map and fam in reas_map[vid] and math.isfinite(val):
            reas_map[vid][fam].append(val)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(REASONING_FAMILIES)))
    for vid, color, shift, label in (
        ("A1_baseline_puro", "#6272a4", -0.24, "A1"),
        ("A4_metrics_legacy", "#50fa7b", 0.0, "A4"),
        ("A5_fractal_full", "#ffb86c", 0.24, "A5"),
    ):
        vals = []
        for fam in REASONING_FAMILIES:
            xs = reas_map.get(vid, {}).get(fam, [])
            vals.append(statistics.mean(xs) if xs else 0.0)
        ax.bar([i + shift for i in x], vals, width=0.22, label=label, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(REASONING_FAMILIES, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Activación por familia de reasoning (A1/A4/A5)")
    ax.grid(alpha=0.2, axis="y")
    ax.legend()
    path = charts_dir / "reasoning_family_activation_A1_A4_A5.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    manifests.append(
        _chart_record(
            out_dir=out_dir,
            chart_id="reasoning_family_activation_A1_A4_A5",
            path=path,
            series=reas_map,
        )
    )

    # 5) Bridge vs metrics effect
    labels = ["A2-A1", "A5-A4", "A5-A1"]
    keys = ["A2_vs_A1", "A5_vs_A4", "A5_vs_A1"]
    fhs = [_safe_float(effects.get(k, {}).get("FHS_delta", {}).get("point")) for k in keys]
    sss = [_safe_float(effects.get(k, {}).get("SSS_delta", {}).get("point")) for k in keys]
    ops = [_safe_float(effects.get(k, {}).get("OPS_delta", {}).get("point")) for k in keys]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.24
    ax.bar([i - w for i in x], fhs, width=w, label="ΔFHS")
    ax.bar(x, sss, width=w, label="ΔSSS")
    ax.bar([i + w for i in x], ops, width=w, label="ΔOPS")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Comparación causal: bridge vs métricas")
    ax.grid(alpha=0.2, axis="y")
    ax.legend()
    path = charts_dir / "bridge_effect_vs_metrics_effect.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    manifests.append(
        _chart_record(
            out_dir=out_dir,
            chart_id="bridge_effect_vs_metrics_effect",
            path=path,
            series={"labels": labels, "delta_FHS": fhs, "delta_SSS": sss, "delta_OPS": ops},
        )
    )

    return manifests


def write_sensitivity_report(path: Path, sensitivity: dict[str, Any]) -> None:
    lines = ["# Sensitivity Report", ""]
    lines.append(f"- status: {sensitivity.get('status')}")
    lines.append(f"- shared_seeds: {sensitivity.get('shared_seeds')}")
    summary = sensitivity.get("summary", {})
    if summary:
        lines.append(f"- stable_positive_fraction: {float(summary.get('stable_positive_fraction', 0.0)):.3f}")
        lines.append(f"- has_large_negative: {summary.get('has_large_negative')}")
    lines.append("")
    lines.append("## Leave-one-seed-out")
    for row in sensitivity.get("leave_one_seed_out", []):
        lines.append(
            f"- seed={row['left_out_seed']}: Δ={float(row['delta_point']):.3f}, "
            f"CI95 [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}], "
            f"p_perm={float(row['p_perm']):.5f}"
        )
    lines.append("")
    lines.append("## Leave-one-probe-out")
    for row in sensitivity.get("leave_one_probe_out", []):
        lines.append(
            f"- probe={row['excluded_probe']}: Δ={float(row['delta_point']):.3f}, "
            f"CI95 [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}], "
            f"p_perm={float(row['p_perm']):.5f}"
        )
    lines.append("")
    lines.append("## Leave-one-metric-family-out")
    for row in sensitivity.get("leave_one_metric_family_out", []):
        lines.append(
            f"- family={row['excluded_family']}: Δ={float(row['delta_point']):.3f}, "
            f"CI95 [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}], "
            f"p_perm={float(row['p_perm']):.5f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(
    *,
    out_path: Path,
    variant_rows: list[dict[str, Any]],
    category: str,
    rationale: str,
    effects: dict[str, dict[str, Any]],
    sensitivity: dict[str, Any],
    negative_controls: dict[str, Any],
    family_summary_rows: list[dict[str, Any]],
    family_diagnostics: dict[str, Any],
) -> None:
    by_id = {r["variant_id"]: r for r in variant_rows}
    lines: list[str] = []
    lines.append("# Fractal vs No Fractal - Estudio Causal 5x")
    lines.append("")
    lines.append("## Veredicto científico")
    lines.append(f"- Clasificación final: **{category}**")
    lines.append(f"- Justificación breve: {rationale}")
    lines.append("")
    a1 = by_id.get("A1_baseline_puro")
    a5 = by_id.get("A5_fractal_full")
    net = effects.get("A5_vs_A1")

    if a1 is not None and a5 is not None and net is not None:
        lines.append("## Resumen control vs tratamiento")
        lines.append(f"- A1 (control): FHS={a1['FHS']:.2f}, SSS={a1['SSS']:.2f}, OPS={a1['OPS']:.2f}")
        lines.append(f"- A5 (tratamiento full): FHS={a5['FHS']:.2f}, SSS={a5['SSS']:.2f}, OPS={a5['OPS']:.2f}")
        lines.append(
            f"- ΔSSS A5-A1: {net['SSS_delta']['point']:.3f} "
            f"(CI95 [{net['SSS_delta']['ci95_low']:.3f}, {net['SSS_delta']['ci95_high']:.3f}])"
        )
        lines.append(
            f"- ΔFHS A5-A1: {net['FHS_delta']['point']:.3f} "
            f"(CI95 [{net['FHS_delta']['ci95_low']:.3f}, {net['FHS_delta']['ci95_high']:.3f}])"
        )
        lines.append(
            f"- ΔOPS A5-A1: {net['OPS_delta']['point']:.3f} "
            f"(CI95 [{net['OPS_delta']['ci95_low']:.3f}, {net['OPS_delta']['ci95_high']:.3f}])"
        )
        lines.append("")
        lines.append("## Señales funcionales críticas")
        lines.append(
            f"- Regresión funcional común (A5): {a5['functional_regression']}"
        )
        lines.append(
            f"- Falsos sub-threshold (A5 promedio): {a5['false_subthreshold_mean']:.3f}"
        )
        lines.append(
            f"- Missing core cau/ctf rate (A5): {a5['missing_core_rate_mean']:.5f}"
        )
    else:
        lines.append("## Resumen de variantes ejecutadas")
        lines.append("- Corrida parcial: no incluye simultáneamente A1 y A5.")
        for row in variant_rows:
            lines.append(
                f"- {row['variant_id']}: FHS={row['FHS']:.2f}, SSS={row['SSS']:.2f}, "
                f"OPS={row['OPS']:.2f}, regression={row['functional_regression']}"
            )
    lines.append("")
    lines.append("## Efectos causales mínimos")
    for key in ("A2_vs_A1", "A3_vs_A1", "A5_vs_A4", "A5_vs_A1"):
        if key not in effects:
            continue
        eff = effects[key]
        g = eff.get("SSS_hedges_g", {})
        cliff = eff.get("SSS_cliffs_delta", {})
        perm = eff.get("SSS_permutation", {})
        lines.append(
            f"- {key}: ΔFHS={eff['FHS_delta']['point']:.3f}, "
            f"ΔSSS={eff['SSS_delta']['point']:.3f}, ΔOPS={eff['OPS_delta']['point']:.3f}, "
            f"Hedges g(SSS)={float(g.get('value', float('nan'))):.3f}, "
            f"CliffΔ(SSS)={float(cliff.get('value', float('nan'))):.3f}, "
            f"p_perm(SSS)={float(perm.get('p_two_sided', float('nan'))):.5f}"
        )
        diag = eff.get("diagnostics", {}).get("SSS", {})
        if diag:
            lines.append(
                f"  diagnóstico: n={diag.get('n_pairs')}, "
                f"ci_degenerate={diag.get('ci_degenerate')}, "
                f"g_undefined={diag.get('hedges_g_undefined')}, "
                f"sign_consistent={diag.get('sign_consistent')}"
            )
    lines.append("")
    lines.append("## Detalle por familias")
    lines.append("- Cobertura geométrica esperada: 12 familias (T,C,F3D,B,AXC,AXD,MC,RS,AC,GF,W,PF).")
    focus_ids = {"A1_baseline_puro", "A4_metrics_legacy", "A5_fractal_full"}
    for row in variant_rows:
        if row["variant_id"] not in focus_ids:
            continue
        missing = row.get("missing_geometry_families", [])
        lines.append(
            f"- {row['variant_id']}: coverage={100.0 * float(row.get('geometry_family_coverage_ratio', 0.0)):.1f}% "
            f"| missing={missing if missing else '[]'}"
        )

    geom_overall: dict[str, dict[str, float]] = {}
    reas_act: dict[str, dict[str, float]] = {}
    for item in family_summary_rows:
        vid = str(item.get("variant_id", ""))
        if vid not in focus_ids:
            continue
        fam_type = str(item.get("family_type", ""))
        fam_id = str(item.get("family_id", ""))
        metric = str(item.get("metric", ""))
        mean = _safe_float(item.get("mean"))
        if fam_type == "geometry" and metric == "atlas_overall" and math.isfinite(mean):
            geom_overall.setdefault(vid, {})[fam_id] = mean
        if fam_type == "reasoning" and metric == "activation_rate" and math.isfinite(mean):
            reas_act.setdefault(vid, {})[fam_id] = mean

    def _top_items(mapping: dict[str, float], n: int = 3) -> list[tuple[str, float]]:
        return sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)[:n]

    for vid in ("A1_baseline_puro", "A4_metrics_legacy", "A5_fractal_full"):
        if vid in geom_overall:
            top = _top_items(geom_overall[vid], n=3)
            lines.append(f"- {vid} top geometrías por atlas_overall: {top}")
        if vid in reas_act:
            top = _top_items(reas_act[vid], n=4)
            lines.append(f"- {vid} top activación reasoning: {top}")

    lines.append("")
    lines.append("## Respuesta interpretada obligatoria")
    bridge_ops = _safe_float(effects.get("A2_vs_A1", {}).get("OPS_delta", {}).get("point"))
    metrics_ops = _safe_float(effects.get("A5_vs_A4", {}).get("OPS_delta", {}).get("point"))
    net_ops = _safe_float(effects.get("A5_vs_A1", {}).get("OPS_delta", {}).get("point"))
    source_ops = "indeterminado"
    if math.isfinite(bridge_ops) and math.isfinite(metrics_ops):
        source_ops = "bridge" if bridge_ops < metrics_ops else "metrics/policy"
    lines.append(
        f"- Fuente principal de caída de OPS: {source_ops} "
        f"(A2-A1={bridge_ops:.3f}, A5-A4={metrics_ops:.3f}, A5-A1={net_ops:.3f})."
    )

    probe_rows = sensitivity.get("leave_one_probe_out", [])
    net_sss = _safe_float(effects.get("A5_vs_A1", {}).get("SSS_delta", {}).get("point"))
    probe_sensitivity: list[tuple[str, float]] = []
    for item in probe_rows:
        d = _safe_float(item.get("delta_point"))
        if math.isfinite(d):
            probe_sensitivity.append((str(item.get("excluded_probe")), abs(net_sss - d)))
    probe_sensitivity.sort(key=lambda kv: kv[1], reverse=True)
    discriminative = probe_sensitivity[0][0] if probe_sensitivity else "N/A"
    replicating: list[str] = []
    for probe, impact in sorted(probe_sensitivity, key=lambda kv: kv[1]):
        if len(replicating) >= 3:
            break
        replicating.append(probe)
    if negative_controls.get("replicates_effect") is False:
        replicating = []
    lines.append(
        f"- Familias/probes que replican controles negativos: {replicating if replicating else 'ninguno detectado'}."
    )
    lines.append(f"- Subscore/probe con mayor poder discriminativo: {discriminative}.")
    lines.append(f"- Estado final de clasificación: {category}.")

    lines.append("")
    lines.append("## Robustez y controles")
    ssum = sensitivity.get("summary", {})
    lines.append(
        f"- Sensibilidad status={sensitivity.get('status')}, "
        f"stable_positive_fraction={float(ssum.get('stable_positive_fraction', 0.0)):.3f}, "
        f"has_large_negative={ssum.get('has_large_negative')}"
    )
    lines.append(
        f"- Controles negativos status={negative_controls.get('status')}, "
        f"replicates_effect={negative_controls.get('replicates_effect')}"
    )
    if negative_controls.get("status") == "ok":
        sham = negative_controls.get("sham_fractal", {})
        rnd = negative_controls.get("randomized_fractal", {})
        lines.append(
            f"- Sham fractal: Δ={float(sham.get('delta', {}).get('point', float('nan'))):.3f}, "
            f"CI95 [{float(sham.get('delta', {}).get('ci95_low', float('nan'))):.3f}, {float(sham.get('delta', {}).get('ci95_high', float('nan'))):.3f}], "
            f"p_perm={float(sham.get('permutation', {}).get('p_two_sided', float('nan'))):.5f}"
        )
        lines.append(
            f"- Fractal randomizado: Δ={float(rnd.get('delta', {}).get('point', float('nan'))):.3f}, "
            f"CI95 [{float(rnd.get('delta', {}).get('ci95_low', float('nan'))):.3f}, {float(rnd.get('delta', {}).get('ci95_high', float('nan'))):.3f}], "
            f"p_perm={float(rnd.get('permutation', {}).get('p_two_sided', float('nan'))):.5f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validity_risks(
    *,
    out_path: Path,
    category: str,
    variant_rows: list[dict[str, Any]],
) -> None:
    by_id = {r["variant_id"]: r for r in variant_rows}
    a5 = by_id.get("A5_fractal_full")
    risks: list[str] = []
    risks.append("# Validez y riesgos residuales")
    risks.append("")
    risks.append("## Amenazas a validez interna")
    risks.append("- El componente budgeting no se ablaciona de forma directa; su efecto se infiere como residual causal.")
    risks.append("- Algunas métricas estructurales son altamente deterministas; estabilidad alta puede ocultar sensibilidad real.")
    risks.append("- W3 bridge-off usa perfil no-fractal sintético; reduce contaminación, pero no representa todas las cargas posibles.")
    risks.append("")
    risks.append("## Amenazas a validez externa")
    risks.append("- Resultados ligados a esta máquina/entorno; falta confirmación cruzada en hardware distinto.")
    risks.append("- Corpus geométrico y probes actuales cubren familias principales, no el espacio total de geometrías.")
    risks.append("")
    risks.append("## Sensibilidad seeds y umbrales")
    risks.append("- Seeds fijadas y orden fijo (5x) para comparabilidad; recomendable repetir bloque con set alterno de seeds.")
    risks.append("- Umbrales base congelados (0.7/0.45/0.4) durante todo el estudio.")
    risks.append("")
    risks.append("## Riesgo de degeneración de métricas")
    if a5 is not None:
        risks.append(
            f"- Dominancia de avalanchas en A5: {a5['avalanche_dominant_fraction_mean']:.3f} (menor es mejor)."
        )
        risks.append(
            f"- Error temporal medio A5: {a5['temporal_scale_error_mean']:.3f}."
        )
    else:
        risks.append("- A5 no disponible en esta corrida parcial; no se puede cuantificar degeneración específica.")
    risks.append("")
    risks.append("## Estado final")
    risks.append(f"- Clasificación final observada: **{category}**")
    out_path.write_text("\n".join(risks) + "\n", encoding="utf-8")


def validate_ref(repo_root: Path, ref: str) -> None:
    proc = run_cmd(["git", "rev-parse", "--verify", ref], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(f"Referencia git inválida: {ref}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    seeds = parse_seeds(args.seeds)
    variants = resolve_variants(args.variants)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_root / ts
    run_id = f"ablation5x-{ts}-{uuid.uuid4().hex[:8]}"
    variants_root = out_dir / "variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_root.mkdir(parents=True, exist_ok=True)

    validate_ref(repo_root, args.stable_ref)
    validate_ref(repo_root, args.legacy_ref)

    report: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": ts,
        "repo_root": str(repo_root),
        "out_dir": str(out_dir),
        "stable_ref": args.stable_ref,
        "legacy_ref": args.legacy_ref,
        "seeds": seeds,
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "workload_scale": args.workload_scale,
        "variants": {},
    }

    total_steps = len(variants) * (
        1
        + (0 if args.skip_suites else 4)
        + (0 if args.skip_probes else (len(seeds) * 3))
        + (0 if args.skip_benchmarks else (len(seeds) * 3))
    )
    done_steps = 0
    started = time.monotonic()

    def progress(label: str) -> None:
        nonlocal done_steps
        done_steps += 1
        elapsed = time.monotonic() - started
        eta = ((elapsed / done_steps) * (total_steps - done_steps)) if done_steps > 0 else 0.0
        pct = (100.0 * done_steps / max(1, total_steps))
        eprint(f"[{pct:6.2f}%] {done_steps}/{total_steps} | {label} | eta={eta:0.1f}s")

    for variant in variants:
        variant_dir = variants_root / variant.variant_id
        eprint(f"Materializando {variant.variant_id} en {variant_dir}")
        materialize_variant_root(
            repo_root=repo_root,
            stable_ref=args.stable_ref,
            legacy_ref=args.legacy_ref,
            variant=variant,
            variant_root=variant_dir,
        )
        progress(f"{variant.variant_id}:materialize")

        v_report: dict[str, Any] = {
            "meta": {
                "variant_id": variant.variant_id,
                "label": variant.label,
                "policy_ref": variant.policy_ref,
                "bridge_ref": variant.bridge_ref,
                "bridge_mode": variant.bridge_mode,
                "metrics_ref": variant.metrics_ref,
                "root": str(variant_dir),
            },
            "suites": {},
            "functional_probe_runs": [],
            "structural_probe_runs": [],
            "family_probe_runs": [],
            "benchmarks": {"W1": [], "W2": [], "W3": []},
        }

        suite_out = out_dir / "runs" / variant.variant_id / "suites"
        probe_out = out_dir / "runs" / variant.variant_id / "probes"
        bench_out = out_dir / "runs" / variant.variant_id / "bench"
        suite_out.mkdir(parents=True, exist_ok=True)
        probe_out.mkdir(parents=True, exist_ok=True)
        bench_out.mkdir(parents=True, exist_ok=True)

        if not args.skip_suites:
            v_report["suites"]["common_regression"] = run_pytest_suite(
                suite_name="common_regression",
                tests_expr=COMMON_REGRESSION,
                root=variant_dir,
                out_dir=suite_out,
            )
            progress(f"{variant.variant_id}:suite:common_regression")

            v_report["suites"]["common_stress"] = run_pytest_suite(
                suite_name="common_stress",
                tests_expr=COMMON_STRESS,
                root=variant_dir,
                out_dir=suite_out,
            )
            progress(f"{variant.variant_id}:suite:common_stress")

            v_report["suites"]["ordered_suite"] = run_pytest_suite(
                suite_name="ordered_suite",
                tests_expr=ORDERED_SUITE,
                root=variant_dir,
                out_dir=suite_out,
            )
            progress(f"{variant.variant_id}:suite:ordered_suite")

            if variant.bridge_mode == "on":
                v_report["suites"]["fractal_only"] = run_pytest_suite(
                    suite_name="fractal_only",
                    tests_expr=FRACTAL_ONLY_SUITE,
                    root=variant_dir,
                    out_dir=suite_out,
                )
            else:
                v_report["suites"]["fractal_only"] = {
                    "suite": "fractal_only",
                    "status": "N/A (bridge off)",
                }

            v_report["suites"]["structural_probes"] = run_pytest_suite(
                suite_name="structural_probes",
                tests_expr=STRUCTURAL_PROBES,
                root=variant_dir,
                out_dir=suite_out,
            )
            progress(f"{variant.variant_id}:suite:structural_probes")

        if not args.skip_probes:
            for seed in seeds:
                rc, metrics = run_inline_python(
                    code=functional_probe_code(seed),
                    root=variant_dir,
                    log_path=probe_out / f"functional_seed{seed}.log",
                )
                if rc == 0 and metrics is not None:
                    v_report["functional_probe_runs"].append(metrics)
                progress(f"{variant.variant_id}:probe:functional:seed={seed}")

            for seed in seeds:
                rc, metrics = run_inline_python(
                    code=structural_probe_code(seed),
                    root=variant_dir,
                    log_path=probe_out / f"structural_seed{seed}.log",
                )
                if rc == 0 and metrics is not None:
                    v_report["structural_probe_runs"].append(metrics)
                progress(f"{variant.variant_id}:probe:structural:seed={seed}")

            for seed in seeds:
                rc, metrics = run_inline_python(
                    code=family_probe_code(seed, variant.bridge_mode),
                    root=variant_dir,
                    log_path=probe_out / f"families_seed{seed}.log",
                )
                if rc == 0 and metrics is not None:
                    v_report["family_probe_runs"].append(metrics)
                progress(f"{variant.variant_id}:probe:families:seed={seed}")

            v_report["functional_probe_summary"] = summarize_probe_runs(
                v_report["functional_probe_runs"]
            )
            v_report["structural_probe_summary"] = summarize_structural_runs(
                v_report["structural_probe_runs"]
            )
            v_report["family_probe_summary"] = summarize_family_runs(
                v_report["family_probe_runs"],
                bridge_mode=variant.bridge_mode,
            )
        else:
            v_report["functional_probe_summary"] = {"ok_runs": 0}
            v_report["structural_probe_summary"] = {"ok_runs": 0}
            v_report["family_probe_summary"] = {
                "ok_runs": 0,
                "family_details_rows": [],
                "family_summary_rows": [],
                "family_diagnostics": {
                    "status": "skip_probes",
                    "bridge_mode": variant.bridge_mode,
                    "expected_geometry_families": GEOMETRY_FAMILIES,
                    "observed_geometry_families": [],
                    "missing_geometry_families": GEOMETRY_FAMILIES,
                },
            }

        if not args.skip_benchmarks:
            for seed in seeds:
                v_report["benchmarks"]["W1"].append(
                    run_timed_python(
                        code=workload_w1_code(seed, args.workload_scale),
                        root=variant_dir,
                        log_path=bench_out / f"W1_seed{seed}.log",
                    )
                )
                progress(f"{variant.variant_id}:bench:W1:seed={seed}")

            for seed in seeds:
                v_report["benchmarks"]["W2"].append(
                    run_timed_python(
                        code=workload_w2_code(seed, args.workload_scale),
                        root=variant_dir,
                        log_path=bench_out / f"W2_seed{seed}.log",
                    )
                )
                progress(f"{variant.variant_id}:bench:W2:seed={seed}")

            for seed in seeds:
                code = (
                    workload_w3_fractal_code(seed, args.workload_scale)
                    if variant.bridge_mode == "on"
                    else workload_w3_nonfractal_code(seed, args.workload_scale)
                )
                v_report["benchmarks"]["W3"].append(
                    run_timed_python(
                        code=code,
                        root=variant_dir,
                        log_path=bench_out / f"W3_seed{seed}.log",
                    )
                )
                progress(f"{variant.variant_id}:bench:W3:seed={seed}")

        v_report["benchmarks"]["W1_summary"] = summarize_repeats(v_report["benchmarks"]["W1"])
        v_report["benchmarks"]["W2_summary"] = summarize_repeats(v_report["benchmarks"]["W2"])
        v_report["benchmarks"]["W3_summary"] = summarize_repeats(v_report["benchmarks"]["W3"])

        report["variants"][variant.variant_id] = v_report

    rows: list[dict[str, Any]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    base_report = report["variants"].get("A1_baseline_puro")

    for variant in variants:
        v_report = report["variants"][variant.variant_id]
        scores = calc_variant_scores(
            variant_report=v_report,
            baseline_report=base_report if base_report is not None else None,
            bootstrap_samples=args.bootstrap_samples,
        )
        row = {
            "variant_id": variant.variant_id,
            "label": variant.label,
            "bridge_mode": variant.bridge_mode,
            "policy_ref": variant.policy_ref,
            "bridge_ref": variant.bridge_ref,
            "metrics_ref": variant.metrics_ref,
            **scores,
        }
        rows.append(row)
        rows_by_id[variant.variant_id] = row

    mandatory_ids = {
        "A1_baseline_puro",
        "A2_bridge_on_policy_baseline",
        "A3_policy_modulada_sin_fractal_full",
        "A4_metrics_legacy",
        "A5_fractal_full",
    }
    have_all = mandatory_ids.issubset(set(rows_by_id.keys()))
    effects: dict[str, dict[str, Any]] = {}
    if have_all:
        effects["A2_vs_A1"] = effect_bundle(
            base=rows_by_id["A1_baseline_puro"],
            other=rows_by_id["A2_bridge_on_policy_baseline"],
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=111,
        )
        effects["A3_vs_A1"] = effect_bundle(
            base=rows_by_id["A1_baseline_puro"],
            other=rows_by_id["A3_policy_modulada_sin_fractal_full"],
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=211,
        )
        effects["A5_vs_A4"] = effect_bundle(
            base=rows_by_id["A4_metrics_legacy"],
            other=rows_by_id["A5_fractal_full"],
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=311,
        )
        effects["A5_vs_A1"] = effect_bundle(
            base=rows_by_id["A1_baseline_puro"],
            other=rows_by_id["A5_fractal_full"],
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=411,
        )
    else:
        for key in ("A2_vs_A1", "A3_vs_A1", "A5_vs_A4", "A5_vs_A1"):
            effects[key] = {
                "FHS_delta": {"point": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")},
                "SSS_delta": {"point": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")},
                "OPS_delta": {"point": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")},
                "SSS_hedges_g": {"value": float("nan"), "undefined": True, "reason": "missing"},
                "SSS_cliffs_delta": {"value": float("nan")},
                "SSS_permutation": {"p_two_sided": float("nan"), "method": "missing"},
                "diagnostics": {
                    "FHS": {"n_pairs": 0, "enough_n": False, "ci_degenerate": True},
                    "SSS": {"n_pairs": 0, "enough_n": False, "ci_degenerate": True},
                    "OPS": {"n_pairs": 0, "enough_n": False, "ci_degenerate": True},
                },
            }

    sensitivity: dict[str, Any]
    negative_controls: dict[str, Any]
    if have_all:
        sensitivity = run_sensitivity_analysis(
            a1=rows_by_id["A1_baseline_puro"],
            a5=rows_by_id["A5_fractal_full"],
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
        )
        negative_controls = run_negative_controls(
            a1=rows_by_id["A1_baseline_puro"],
            a5=rows_by_id["A5_fractal_full"],
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
        )
    else:
        sensitivity = {"status": "missing_variants", "summary": {"stable_positive_fraction": 0.0, "has_large_negative": True}}
        negative_controls = {"status": "missing_variants", "replicates_effect": False}

    category, rationale = classify_outcome(
        rows_by_id=rows_by_id,
        effects=effects,
        have_all=have_all,
        sensitivity=sensitivity,
        negative_controls=negative_controls,
    )

    # Causal map with residual (bridge/policy/metrics + interacción residual/budgeting).
    def point(effect_key: str, metric_key: str) -> float:
        return float(effects[effect_key][metric_key]["point"])

    inference_diagnostics = {
        "effects": {
            key: {
                "FHS": eff.get("diagnostics", {}).get("FHS", {}),
                "SSS": eff.get("diagnostics", {}).get("SSS", {}),
                "OPS": eff.get("diagnostics", {}).get("OPS", {}),
                "SSS_permutation": eff.get("SSS_permutation", {}),
                "SSS_hedges_g": eff.get("SSS_hedges_g", {}),
                "SSS_cliffs_delta": eff.get("SSS_cliffs_delta", {}),
            }
            for key, eff in effects.items()
        },
        "quality_flags": {
            "have_all_variants": have_all,
            "a1_a5_min_n_ok": bool(
                effects.get("A5_vs_A1", {}).get("diagnostics", {}).get("SSS", {}).get("enough_n")
            ),
            "a1_a5_ci_degenerate": bool(
                effects.get("A5_vs_A1", {}).get("diagnostics", {}).get("SSS", {}).get("ci_degenerate")
            ),
            "a1_a5_sign_consistent": bool(
                effects.get("A5_vs_A1", {}).get("diagnostics", {}).get("SSS", {}).get("sign_consistent")
            ),
            "sensitivity_status": sensitivity.get("status"),
            "negative_controls_status": negative_controls.get("status"),
            "negative_controls_replicates_effect": negative_controls.get("replicates_effect"),
        },
    }

    causal_map = {
        "effect_decomposition": {
            "bridge_effect_A2_minus_A1": {
                "delta_FHS": point("A2_vs_A1", "FHS_delta"),
                "delta_SSS": point("A2_vs_A1", "SSS_delta"),
                "delta_OPS": point("A2_vs_A1", "OPS_delta"),
            },
            "policy_effect_A3_minus_A1": {
                "delta_FHS": point("A3_vs_A1", "FHS_delta"),
                "delta_SSS": point("A3_vs_A1", "SSS_delta"),
                "delta_OPS": point("A3_vs_A1", "OPS_delta"),
            },
            "metrics_stabilization_A5_minus_A4": {
                "delta_FHS": point("A5_vs_A4", "FHS_delta"),
                "delta_SSS": point("A5_vs_A4", "SSS_delta"),
                "delta_OPS": point("A5_vs_A4", "OPS_delta"),
            },
            "net_fractal_A5_minus_A1": {
                "delta_FHS": point("A5_vs_A1", "FHS_delta"),
                "delta_SSS": point("A5_vs_A1", "SSS_delta"),
                "delta_OPS": point("A5_vs_A1", "OPS_delta"),
            },
            "residual_interaction_budgeting_inferred": {
                "delta_FHS": (
                    point("A5_vs_A1", "FHS_delta")
                    - point("A2_vs_A1", "FHS_delta")
                    - point("A3_vs_A1", "FHS_delta")
                    - point("A5_vs_A4", "FHS_delta")
                ),
                "delta_SSS": (
                    point("A5_vs_A1", "SSS_delta")
                    - point("A2_vs_A1", "SSS_delta")
                    - point("A3_vs_A1", "SSS_delta")
                    - point("A5_vs_A4", "SSS_delta")
                ),
                "delta_OPS": (
                    point("A5_vs_A1", "OPS_delta")
                    - point("A2_vs_A1", "OPS_delta")
                    - point("A3_vs_A1", "OPS_delta")
                    - point("A5_vs_A4", "OPS_delta")
                ),
                "note": "Incluye no linealidad, sinergia y residuo atribuible a budgeting no ablacionado directamente.",
            },
        },
        "classification": {
            "category": category,
            "rationale": rationale,
        },
        "inference_diagnostics": inference_diagnostics["quality_flags"],
    }

    comparison_csv = out_dir / "comparison_table.csv"
    export_comparison_csv(comparison_csv, rows)
    per_seed_csv = out_dir / "per_seed_metrics.csv"
    export_per_seed_metrics_csv(per_seed_csv, rows)
    score_breakdown_csv = out_dir / "score_breakdown.csv"
    export_score_breakdown_csv(score_breakdown_csv, rows)
    family_details_csv = out_dir / "family_details_long.csv"
    family_details_rows = export_family_details_long_csv(family_details_csv, rows)
    family_summary_csv = out_dir / "family_summary.csv"
    family_summary_rows = export_family_summary_csv(family_summary_csv, rows)
    family_diagnostics = collect_family_diagnostics(rows)
    family_diagnostics_json = out_dir / "family_diagnostics.json"
    family_diagnostics_json.write_text(
        json.dumps(family_diagnostics, indent=2),
        encoding="utf-8",
    )

    chart_manifest = generate_comparative_charts(
        out_dir=out_dir,
        rows=rows,
        effects=effects,
        family_details_rows=family_details_rows,
    )
    chart_manifest_json = out_dir / "chart_manifest.json"
    chart_manifest_json.write_text(json.dumps(chart_manifest, indent=2), encoding="utf-8")

    summary_md = out_dir / "summary.md"
    write_summary_md(
        out_path=summary_md,
        variant_rows=rows,
        category=category,
        rationale=rationale,
        effects=effects,
        sensitivity=sensitivity,
        negative_controls=negative_controls,
        family_summary_rows=family_summary_rows,
        family_diagnostics=family_diagnostics,
    )

    validity_md = out_dir / "validity_risks.md"
    write_validity_risks(
        out_path=validity_md,
        category=category,
        variant_rows=rows,
    )

    sensitivity_md = out_dir / "sensitivity_report.md"
    write_sensitivity_report(sensitivity_md, sensitivity)

    negative_controls_json = out_dir / "negative_controls.json"
    negative_controls_json.write_text(json.dumps(negative_controls, indent=2), encoding="utf-8")

    inference_diag_json = out_dir / "inference_diagnostics.json"
    inference_diag_json.write_text(json.dumps(inference_diagnostics, indent=2), encoding="utf-8")

    causal_json = out_dir / "causal_map.json"
    causal_json.write_text(json.dumps(causal_map, indent=2), encoding="utf-8")

    report["analysis"] = {
        "variant_rows": rows,
        "effects": effects,
        "classification": {"category": category, "rationale": rationale},
        "sensitivity": sensitivity,
        "negative_controls": negative_controls,
        "inference_diagnostics": inference_diagnostics,
        "family_diagnostics": family_diagnostics,
        "chart_manifest": chart_manifest,
        "artifacts": {
            "summary_md": str(summary_md),
            "comparison_table_csv": str(comparison_csv),
            "per_seed_metrics_csv": str(per_seed_csv),
            "score_breakdown_csv": str(score_breakdown_csv),
            "family_details_long_csv": str(family_details_csv),
            "family_summary_csv": str(family_summary_csv),
            "family_diagnostics_json": str(family_diagnostics_json),
            "chart_manifest_json": str(chart_manifest_json),
            "causal_map_json": str(causal_json),
            "inference_diagnostics_json": str(inference_diag_json),
            "negative_controls_json": str(negative_controls_json),
            "sensitivity_report_md": str(sensitivity_md),
            "validity_risks_md": str(validity_md),
        },
    }

    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    persist_run_to_sqlite(
        db_path=args.db_path,
        run_id=run_id,
        report=report,
        rows=rows,
        effects=effects,
        inference_diagnostics=inference_diagnostics,
        negative_controls=negative_controls,
        sensitivity=sensitivity,
        category=category,
        rationale=rationale,
        chart_manifest=chart_manifest,
        family_details_rows=family_details_rows,
        family_summary_rows=family_summary_rows,
        family_diagnostics=family_diagnostics,
    )

    if not args.keep_variants:
        shutil.rmtree(variants_root, ignore_errors=True)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "report_json": str(report_json),
                "summary_md": str(summary_md),
                "comparison_table_csv": str(comparison_csv),
                "per_seed_metrics_csv": str(per_seed_csv),
                "score_breakdown_csv": str(score_breakdown_csv),
                "family_details_long_csv": str(family_details_csv),
                "family_summary_csv": str(family_summary_csv),
                "family_diagnostics_json": str(family_diagnostics_json),
                "chart_manifest_json": str(chart_manifest_json),
                "causal_map_json": str(causal_json),
                "inference_diagnostics_json": str(inference_diag_json),
                "negative_controls_json": str(negative_controls_json),
                "sensitivity_report_md": str(sensitivity_md),
                "validity_risks_md": str(validity_md),
                "sqlite_db_path": str(args.db_path),
                "run_id": run_id,
                "classification": category,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

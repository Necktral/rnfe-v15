#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_BASE_DIR = Path('/home/necktral/rnfe.v15')
DEFAULT_FRACTAL_DIR = Path('/tmp/rnfe-fractal-bfcdfbc')
DEFAULT_OUT_ROOT = Path('/tmp/rnfe_bench_reports')
TS = datetime.now().strftime('%Y%m%d-%H%M%S')
OUT_DIR = DEFAULT_OUT_ROOT / TS

ROOTS = {
    'baseline': DEFAULT_BASE_DIR,
    'fractal': DEFAULT_FRACTAL_DIR,
}

COMMON_SUITES = {
    'common_regression': (
        'tests/regression/test_reasoning_meta_pipeline.py '
        'tests/regression/test_meta_scheduler_policy_units.py '
        'tests/regression/test_meta_scheduler_storage_trace.py '
        'tests/regression/test_meta_scheduler_adaptive.py'
    ),
    'common_stress': (
        'tests/reasoning_stress/test_boundary_sweep.py '
        'tests/reasoning_stress/test_pairwise_interaction.py '
        'tests/reasoning_stress/test_hypercube_sampling.py '
        'tests/reasoning_stress/test_adversarial_thresholds.py '
        'tests/reasoning_stress/test_temporal_hysteresis.py '
        'tests/reasoning_stress/test_family_contribution.py '
        'tests/reasoning_stress/test_atlas_comprehensive.py'
    ),
}

FRACTAL_ONLY_SUITE = (
    'tests/reasoning_stress/test_geometry_catalog.py '
    'tests/reasoning_stress/test_experiment2_atlas.py '
    'tests/reasoning_stress/test_fractal_atlas.py '
    'tests/reasoning_stress/test_multiscale_boundary.py '
    'tests/reasoning_stress/test_box_counting.py '
    'tests/reasoning_stress/test_activation_avalanche.py '
    'tests/reasoning_stress/test_temporal_cascade.py'
)

SEEDS = [101, 202, 303, 404, 505]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Batería fractal vs no fractal (full o W3 únicamente).'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'w3'],
        default='full',
        help='`w3` ejecuta solo benchmark fractal W3; `full` ejecuta toda la batería.',
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=DEFAULT_BASE_DIR,
        help='Root baseline/no-fractal.',
    )
    parser.add_argument(
        '--fractal-dir',
        type=Path,
        default=DEFAULT_FRACTAL_DIR,
        help='Root fractal.',
    )
    parser.add_argument(
        '--out-root',
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help='Directorio base para reportes.',
    )
    return parser.parse_args()


def run_cmd(cmd: str, cwd: Path, log_path: Path, extra_env: dict[str, str] | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '0'
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(
        ['bash', '-lc', cmd],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {cmd}\n\n[stdout]\n{p.stdout}\n\n[stderr]\n{p.stderr}\n",
        encoding='utf-8',
    )
    return p.returncode, p.stdout, p.stderr


def parse_junit(xml_path: Path) -> dict[str, Any]:
    if not xml_path.exists():
        return {
            'tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'failures_by_file': {},
        }
    root = ET.parse(xml_path).getroot()
    tests = failures = errors = skipped = 0
    failures_by_file: dict[str, str] = {}

    if root.tag == 'testsuites':
        suites = root.findall('testsuite')
    else:
        suites = [root]

    for ts in suites:
        tests += int(ts.attrib.get('tests', 0))
        failures += int(ts.attrib.get('failures', 0))
        errors += int(ts.attrib.get('errors', 0))
        skipped += int(ts.attrib.get('skipped', 0))
        for tc in ts.findall('testcase'):
            failure_node = tc.find('failure')
            error_node = tc.find('error')
            if failure_node is None and error_node is None:
                continue
            classname = tc.attrib.get('classname', 'unknown')
            file_key = classname.replace('.', '/') + '.py'
            if file_key not in failures_by_file:
                node = failure_node if failure_node is not None else error_node
                msg = (node.attrib.get('message', '') if node is not None else '')
                text = (node.text or '').strip() if node is not None else ''
                snippet = (msg or text or 'failure/error').splitlines()[0][:300]
                failures_by_file[file_key] = f"{tc.attrib.get('name','<unknown>')}: {snippet}"

    passed = max(0, tests - failures - errors - skipped)
    return {
        'tests': tests,
        'passed': passed,
        'failed': failures,
        'errors': errors,
        'skipped': skipped,
        'failures_by_file': failures_by_file,
    }


def extract_first_trace(log_path: Path) -> str:
    if not log_path.exists():
        return ''
    lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
    markers = ['FAILURES', 'ERRORS', 'short test summary info']
    start_idx = None
    for i, line in enumerate(lines):
        if any(m in line for m in markers):
            start_idx = i
            break
    if start_idx is None:
        return ''
    end_idx = min(len(lines), start_idx + 60)
    snippet = lines[start_idx:end_idx]
    return '\n'.join(snippet)


def run_preflight(root_name: str, root_dir: Path) -> dict[str, Any]:
    out_dir = OUT_DIR / root_name / 'preflight'
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight_cmds = {
        'python_version': 'python --version',
        'pytest_version': 'pytest --version',
        'numpy_pytest_check': textwrap.dedent('''\
            python - <<'PY'
            import importlib.util
            print('numpy', bool(importlib.util.find_spec('numpy')))
            print('pytest', bool(importlib.util.find_spec('pytest')))
            PY
        ''').strip(),
        'time_check': '/usr/bin/time -v true',
    }
    data: dict[str, Any] = {}
    for key, cmd in preflight_cmds.items():
        rc, out, err = run_cmd(cmd, root_dir, out_dir / f'{key}.log')
        data[key] = {
            'returncode': rc,
            'stdout': out.strip(),
            'stderr': err.strip(),
        }
    return data


def run_suite(root_name: str, root_dir: Path, suite_name: str, tests_expr: str) -> dict[str, Any]:
    suite_dir = OUT_DIR / root_name / 'suites'
    suite_dir.mkdir(parents=True, exist_ok=True)
    xml_path = suite_dir / f'{suite_name}.xml'
    log_path = suite_dir / f'{suite_name}.log'
    cmd = f"pytest -q {tests_expr} --durations=30 --tb=short --junitxml {xml_path}"
    rc, out, err = run_cmd(cmd, root_dir, log_path)
    junit = parse_junit(xml_path)
    return {
        'suite': suite_name,
        'returncode': rc,
        'command': cmd,
        'junit': junit,
        'first_trace': extract_first_trace(log_path),
        'log_path': str(log_path),
        'xml_path': str(xml_path),
    }


def parse_time_rss_kb(stderr_text: str) -> int | None:
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr_text)
    return int(m.group(1)) if m else None


def run_timed_python(code: str, cwd: Path, log_path: Path) -> dict[str, Any]:
    wrapped = f"/usr/bin/time -v python - <<'PY'\n{code}\nPY"
    rc, out, err = run_cmd(wrapped, cwd, log_path)
    metrics = None
    for line in out.splitlines()[::-1]:
        if line.startswith('METRICS_JSON='):
            metrics = json.loads(line.split('=', 1)[1])
            break
    return {
        'returncode': rc,
        'stdout': out,
        'stderr': err,
        'rss_kb': parse_time_rss_kb(err),
        'metrics': metrics,
    }


def workload_w1_code(seed: int) -> str:
    return textwrap.dedent(f'''\
    import json, random, time
    from runtime.reasoning.scheduler_meta.budgeting import compute_budget
    from runtime.reasoning.scheduler_meta.policy import select_sequence

    N = 1_000_000
    B = 200
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
    ''')


def workload_w2_code(seed: int) -> str:
    return textwrap.dedent(f'''\
    import json, random, time
    from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler

    N = 100_000
    B = 50
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
    ''')


def workload_w3_code(seed: int) -> str:
    return textwrap.dedent(f'''\
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
        ('carpet', lambda s: generate_cantor_carpet(GeometricParameters(family=FractalFamily.CARPET, depth=3, grid_size=(3,3), seed=s))),
        ('menger', lambda s: generate_menger_sponge(GeometricParameters(family=FractalFamily.VOLUMETRIC_3D, depth=2, seed=s))),
        ('tree', lambda s: generate_fractal_tree(GeometricParameters(family=FractalFamily.BRANCHING, branching_factor=2, depth=5, seed=s))[0]),
        ('lorenz', lambda s: generate_lorenz_attractor(GeometricParameters(family=FractalFamily.CONTINUOUS_ATTRACTOR, trajectory_length=2000, seed=s))[::8]),
        ('henon', lambda s: generate_henon_attractor(GeometricParameters(family=FractalFamily.DISCRETE_ATTRACTOR, trajectory_length=2500, seed=s))),
        ('mandelbrot', lambda s: generate_mandelbrot_set(GeometricParameters(family=FractalFamily.COMPLEX_PLANE, resolution=72, depth=50, seed=s))),
        ('life', lambda s: generate_game_of_life_pattern(GeometricParameters(family=FractalFamily.CELLULAR_AUTOMATA, grid_size=(48,48), depth=25, seed=s))),
        ('scale_free', lambda s: generate_scale_free_graph(GeometricParameters(family=FractalFamily.FRACTAL_GRAPH, grid_size=(60,), branching_factor=3, seed=s))[0]),
        ('wavelet', lambda s: np.vstack([np.column_stack([np.linspace(0,1,len(c)), c/(np.max(np.abs(c))+1e-10)]) for _, c in generate_wavelet_decomposition(GeometricParameters(family=FractalFamily.WAVELET, depth=3, resolution=128, seed=s)).items() if len(c)>0])),
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

    N = 120_000
    B = 40
    sched = MetaScheduler(mode='adaptive')
    lat_us = []
    total = 0
    t0_all = time.perf_counter()

    while total < N:
        batch = B if (N - total) >= B else (N - total)
        t0 = time.perf_counter()
        for _ in range(batch):
            f = corpus[(total + _) % len(corpus)]
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
    ''')


def summarize_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in repeats if r.get('metrics')]
    if not ok:
        return {'ok_runs': 0}
    keys = ['elapsed_s', 'iters_per_sec', 'p50_us', 'p95_us']
    out = {'ok_runs': len(ok)}
    for k in keys:
        vals = [float(r['metrics'][k]) for r in ok]
        out[k] = {
            'mean': statistics.mean(vals),
            'stdev': statistics.pstdev(vals),
            'values': vals,
        }
    rss_vals = [int(r['rss_kb']) for r in ok if r.get('rss_kb') is not None]
    if rss_vals:
        out['rss_kb'] = {
            'mean': statistics.mean(rss_vals),
            'stdev': statistics.pstdev(rss_vals),
            'values': rss_vals,
        }
    return out


def bootstrap_ci_delta_pct(base_vals: list[float], frac_vals: list[float], n_boot: int = 10000, seed: int = 1337) -> dict[str, float]:
    rng = random.Random(seed)
    if not base_vals or not frac_vals:
        return {'point_pct': float('nan'), 'ci95_low': float('nan'), 'ci95_high': float('nan')}
    point = ((statistics.mean(frac_vals) - statistics.mean(base_vals)) / statistics.mean(base_vals)) * 100.0
    boots = []
    n0, n1 = len(base_vals), len(frac_vals)
    for _ in range(n_boot):
        s0 = [base_vals[rng.randrange(n0)] for _ in range(n0)]
        s1 = [frac_vals[rng.randrange(n1)] for _ in range(n1)]
        m0 = statistics.mean(s0)
        m1 = statistics.mean(s1)
        boots.append(((m1 - m0) / m0) * 100.0)
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return {'point_pct': point, 'ci95_low': lo, 'ci95_high': hi}


def main() -> int:
    args = parse_args()
    global OUT_DIR, ROOTS
    OUT_DIR = args.out_root / TS
    ROOTS = {
        'baseline': args.base_dir.resolve(),
        'fractal': args.fractal_dir.resolve(),
    }

    if not ROOTS['fractal'].exists():
        print(f"Fractal root no existe: {ROOTS['fractal']}", file=sys.stderr)
        return 2
    if args.mode == 'full' and not ROOTS['baseline'].exists():
        print(f"Baseline root no existe: {ROOTS['baseline']}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == 'w3':
        report: dict[str, Any] = {
            'timestamp': TS,
            'mode': 'w3',
            'out_dir': str(OUT_DIR),
            'roots': {'fractal': str(ROOTS['fractal'])},
            'benchmarks': {'fractal': {'W3': []}},
            'analysis': {},
        }
        bdir = OUT_DIR / 'fractal' / 'bench'
        bdir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            result = run_timed_python(
                workload_w3_code(seed),
                ROOTS['fractal'],
                bdir / f'W3_seed{seed}.log',
            )
            report['benchmarks']['fractal']['W3'].append(result)

        summary = summarize_repeats(report['benchmarks']['fractal']['W3'])
        ok_runs = int(summary.get('ok_runs', 0))
        status = 'PASS' if ok_runs == len(SEEDS) else 'FAIL'
        report['analysis'] = {
            'benchmark_summaries': {'fractal': {'W3': summary}},
            'ok_runs': ok_runs,
            'required_runs': len(SEEDS),
            'status': status,
        }

        report_json_path = OUT_DIR / 'report.json'
        report_json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        report_md_path = OUT_DIR / 'report.md'
        md = [
            '# W3 Benchmark Report',
            '',
            f"- Timestamp: `{TS}`",
            f"- Fractal root: `{ROOTS['fractal']}`",
            f"- ok_runs: `{ok_runs}/{len(SEEDS)}`",
            f"- status: **{status}**",
        ]
        if summary.get('ok_runs'):
            md.append(
                f"- W3 mean: iters/s={summary['iters_per_sec']['mean']:.2f}, "
                f"p50_us={summary['p50_us']['mean']:.3f}, "
                f"p95_us={summary['p95_us']['mean']:.3f}, "
                f"rss_kb={summary.get('rss_kb', {}).get('mean', float('nan')):.0f}"
            )
        report_md_path.write_text('\n'.join(md), encoding='utf-8')
        print(json.dumps({
            'out_dir': str(OUT_DIR),
            'report_json': str(report_json_path),
            'report_md': str(report_md_path),
            'status': status,
            'ok_runs': ok_runs,
        }, indent=2))
        return 0 if status == 'PASS' else 1

    report: dict[str, Any] = {
        'timestamp': TS,
        'mode': 'full',
        'out_dir': str(OUT_DIR),
        'roots': {k: str(v) for k, v in ROOTS.items()},
        'preflight': {},
        'suites': {},
        'benchmarks': {},
        'analysis': {},
    }

    for root_name, root_dir in ROOTS.items():
        report['preflight'][root_name] = run_preflight(root_name, root_dir)

    for root_name, root_dir in ROOTS.items():
        report['suites'][root_name] = {}
        for suite_name, tests_expr in COMMON_SUITES.items():
            report['suites'][root_name][suite_name] = run_suite(root_name, root_dir, suite_name, tests_expr)

    report['suites']['fractal']['fractal_only'] = run_suite(
        'fractal',
        ROOTS['fractal'],
        'fractal_only',
        FRACTAL_ONLY_SUITE,
    )
    report['suites']['baseline']['fractal_only'] = {
        'suite': 'fractal_only',
        'status': 'N/A (not present in baseline local)',
    }

    for root_name, root_dir in ROOTS.items():
        bdir = OUT_DIR / root_name / 'bench'
        bdir.mkdir(parents=True, exist_ok=True)
        report['benchmarks'][root_name] = {'W1': [], 'W2': [], 'W3': []}
        for seed in SEEDS:
            report['benchmarks'][root_name]['W1'].append(
                run_timed_python(workload_w1_code(seed), root_dir, bdir / f'W1_seed{seed}.log')
            )
            report['benchmarks'][root_name]['W2'].append(
                run_timed_python(workload_w2_code(seed), root_dir, bdir / f'W2_seed{seed}.log')
            )

    for seed in SEEDS:
        report['benchmarks']['fractal']['W3'].append(
            run_timed_python(
                workload_w3_code(seed),
                ROOTS['fractal'],
                OUT_DIR / 'fractal' / 'bench' / f'W3_seed{seed}.log',
            )
        )

    report['benchmarks']['baseline']['W3'] = [{'status': 'N/A (fractal workload only)'}]

    summaries = {
        'baseline': {
            'W1': summarize_repeats(report['benchmarks']['baseline']['W1']),
            'W2': summarize_repeats(report['benchmarks']['baseline']['W2']),
        },
        'fractal': {
            'W1': summarize_repeats(report['benchmarks']['fractal']['W1']),
            'W2': summarize_repeats(report['benchmarks']['fractal']['W2']),
            'W3': summarize_repeats(report['benchmarks']['fractal']['W3']),
        },
    }
    report['analysis']['benchmark_summaries'] = summaries

    def common_totals(root_key: str) -> tuple[int, int, int, int]:
        suites = report['suites'][root_key]
        tests = passed = failed = errors = 0
        for name in ('common_regression', 'common_stress'):
            j = suites[name]['junit']
            tests += j['tests']
            passed += j['passed']
            failed += j['failed']
            errors += j['errors']
        return tests, passed, failed, errors

    b_tests, b_passed, b_failed, b_errors = common_totals('baseline')
    f_tests, f_passed, f_failed, f_errors = common_totals('fractal')

    b_rate = (b_passed / b_tests) if b_tests else 0.0
    f_rate = (f_passed / f_tests) if f_tests else 0.0
    pass_drop_pp = (b_rate - f_rate) * 100.0

    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for workload in ('W1', 'W2'):
        deltas[workload] = {}
        for metric in ('iters_per_sec', 'p95_us', 'rss_kb'):
            bvals = summaries['baseline'][workload].get(metric, {}).get('values', [])
            fvals = summaries['fractal'][workload].get(metric, {}).get('values', [])
            if metric == 'rss_kb':
                bvals = [float(v) for v in bvals]
                fvals = [float(v) for v in fvals]
            deltas[workload][metric] = bootstrap_ci_delta_pct(bvals, fvals)

    report['analysis']['common_pass_rate'] = {
        'baseline': b_rate,
        'fractal': f_rate,
        'drop_pp': pass_drop_pp,
        'baseline_counts': {'tests': b_tests, 'passed': b_passed, 'failed': b_failed, 'errors': b_errors},
        'fractal_counts': {'tests': f_tests, 'passed': f_passed, 'failed': f_failed, 'errors': f_errors},
    }
    report['analysis']['deltas_pct'] = deltas

    p95_slowdown = max(deltas['W1']['p95_us']['point_pct'], deltas['W2']['p95_us']['point_pct'])
    rss_increase = max(deltas['W1']['rss_kb']['point_pct'], deltas['W2']['rss_kb']['point_pct'])
    severe_functional_regression = pass_drop_pp > 10.0 or (f_errors > b_errors + 5)
    if (pass_drop_pp <= 2.0) and (p95_slowdown <= 10.0) and (rss_increase <= 20.0):
        verdict = 'Comparable'
    elif (not severe_functional_regression) and (p95_slowdown <= 15.0) and (rss_increase <= 30.0) and (pass_drop_pp <= 10.0):
        verdict = 'Degradación moderada'
    else:
        verdict = 'No eficiente'

    report['analysis']['efficiency_verdict'] = {
        'verdict': verdict,
        'pass_drop_pp': pass_drop_pp,
        'p95_slowdown_pct_worst': p95_slowdown,
        'rss_increase_pct_worst': rss_increase,
        'severe_functional_regression': severe_functional_regression,
    }

    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    for root in ('baseline', 'fractal'):
        for sname in ('common_regression', 'common_stress'):
            j = report['suites'][root][sname]['junit']
            total_tests += j['tests']
            total_passed += j['passed']
            total_failed += j['failed']
            total_errors += j['errors']
    jf = report['suites']['fractal']['fractal_only']['junit']
    total_tests += jf['tests']
    total_passed += jf['passed']
    total_failed += jf['failed']
    total_errors += jf['errors']
    report['analysis']['totals'] = {
        'tests': total_tests,
        'passed': total_passed,
        'failed': total_failed,
        'errors': total_errors,
    }

    report_json_path = OUT_DIR / 'report.json'
    report_json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    def suite_line(root: str, name: str) -> str:
        s = report['suites'][root][name]
        if 'junit' not in s:
            return f"- `{root}/{name}`: {s.get('status','N/A')}"
        j = s['junit']
        return (
            f"- `{root}/{name}`: tests={j['tests']}, passed={j['passed']}, "
            f"failed={j['failed']}, errors={j['errors']}, skipped={j['skipped']}, rc={s['returncode']}"
        )

    md: list[str] = []
    md.append('# Reporte Fractal vs No Fractal')
    md.append('')
    md.append(f"- Timestamp: `{TS}`")
    md.append(f"- Baseline root: `{ROOTS['baseline']}`")
    md.append(f"- Fractal root: `{ROOTS['fractal']}`")
    md.append('')
    md.append('## Resumen de Suites')
    md.append(suite_line('baseline', 'common_regression'))
    md.append(suite_line('fractal', 'common_regression'))
    md.append(suite_line('baseline', 'common_stress'))
    md.append(suite_line('fractal', 'common_stress'))
    md.append(suite_line('fractal', 'fractal_only'))
    md.append(suite_line('baseline', 'fractal_only'))
    md.append('')
    md.append('## Primera Traza Corta por Fase')
    for root in ('baseline', 'fractal'):
        for phase in ('common_regression', 'common_stress'):
            tr = report['suites'][root][phase].get('first_trace', '')
            md.append(f"### {root}/{phase}")
            md.append('```text')
            md.append((tr[:3000] if tr else 'Sin traza de fallo (suite en verde).'))
            md.append('```')
    tr = report['suites']['fractal']['fractal_only'].get('first_trace', '')
    md.append('### fractal/fractal_only')
    md.append('```text')
    md.append((tr[:3000] if tr else 'Sin traza de fallo (suite en verde).'))
    md.append('```')
    md.append('## Primer fallo/error por archivo (suites)')
    for root in ('baseline', 'fractal'):
        for phase in ('common_regression', 'common_stress'):
            byf = report['suites'][root][phase]['junit'].get('failures_by_file', {})
            md.append(f"### {root}/{phase}")
            if not byf:
                md.append('- Sin fallos/errores por archivo.')
            else:
                for k, v in byf.items():
                    md.append(f"- `{k}` -> {v}")
    byf = report['suites']['fractal']['fractal_only']['junit'].get('failures_by_file', {})
    md.append('### fractal/fractal_only')
    if not byf:
        md.append('- Sin fallos/errores por archivo.')
    else:
        for k, v in byf.items():
            md.append(f"- `{k}` -> {v}")
    md.append('')
    md.append('## Benchmarks (Media de 5 repeticiones)')
    for root in ('baseline', 'fractal'):
        md.append(f"### {root}")
        for w in ('W1', 'W2'):
            s = summaries[root][w]
            if not s.get('ok_runs'):
                md.append(f"- {w}: sin corridas válidas")
                continue
            md.append(
                f"- {w}: iters/s={s['iters_per_sec']['mean']:.2f}, p50_us={s['p50_us']['mean']:.3f}, "
                f"p95_us={s['p95_us']['mean']:.3f}, rss_kb={s.get('rss_kb',{}).get('mean', float('nan')):.0f}"
            )
        if root == 'fractal':
            s = summaries['fractal'].get('W3', {})
            if s.get('ok_runs'):
                md.append(
                    f"- W3: iters/s={s['iters_per_sec']['mean']:.2f}, p50_us={s['p50_us']['mean']:.3f}, "
                    f"p95_us={s['p95_us']['mean']:.3f}, rss_kb={s.get('rss_kb',{}).get('mean', float('nan')):.0f}"
                )
            else:
                md.append('- W3: sin corridas válidas')
    md.append('')
    md.append('## Deltas y CI 95% (Fractal vs Baseline)')
    for w in ('W1', 'W2'):
        md.append(f"### {w}")
        for metric in ('iters_per_sec', 'p95_us', 'rss_kb'):
            d = deltas[w][metric]
            md.append(
                f"- {metric}: point={d['point_pct']:.2f}% | CI95=[{d['ci95_low']:.2f}%, {d['ci95_high']:.2f}%]"
            )
    md.append('')
    md.append('## Veredicto')
    ev = report['analysis']['efficiency_verdict']
    md.append(f"- Veredicto: **{ev['verdict']}**")
    md.append(f"- Pass-rate drop común: {ev['pass_drop_pp']:.2f} pp")
    md.append(f"- Peor p95 slowdown (W1/W2): {ev['p95_slowdown_pct_worst']:.2f}%")
    md.append(f"- Peor RSS increase (W1/W2): {ev['rss_increase_pct_worst']:.2f}%")
    md.append(f"- Regresión funcional severa: {ev['severe_functional_regression']}")
    md.append('')
    totals = report['analysis']['totals']
    md.append('## Totales Ejecutados')
    md.append(
        f"- tests={totals['tests']}, passed={totals['passed']}, failed={totals['failed']}, errors={totals['errors']}"
    )

    report_md_path = OUT_DIR / 'report.md'
    report_md_path.write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({
        'out_dir': str(OUT_DIR),
        'report_json': str(report_json_path),
        'report_md': str(report_md_path),
        'verdict': ev['verdict'],
        'totals': totals,
    }, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

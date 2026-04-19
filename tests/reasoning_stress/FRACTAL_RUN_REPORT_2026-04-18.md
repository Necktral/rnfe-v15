# Informe de Corridas Fractal vs No-Fractal (Fases A/B/C)

**Fecha**: 2026-04-18  
**Rama local**: `work/t5-reapply-20260417-231437`  
**Referencia fractal evaluada**: `origin/claude/integrate-fractal-scheduler-testing` (`6ab844e`)

## Resumen Ejecutivo

| Fase | Estado | Resultado clave |
|---|---|---|
| Fase A | PASS | 71/71 tests en verde (0 fail, 0 error) |
| Fase B | PASS | W3 completo 5/5 (`status=PASS`, `ok_runs=5`) |
| Fase C | FAIL (gate) | Veredicto runner: `No eficiente` por regresión funcional severa |

## Evidencia y Artefactos

- Run root consolidado A/B/C: `/tmp/rnfe_bench_reports/20260418-225836`
- Fase A:
  - Log: `/tmp/rnfe_bench_reports/20260418-225836/phase_a/phase_a.log`
  - JUnit: `/tmp/rnfe_bench_reports/20260418-225836/phase_a/phase_a.xml`
- Fase B:
  - Output JSON: `/tmp/rnfe_bench_reports/20260418-225836/phase_b.stdout.json`
  - Reporte: `/tmp/rnfe_bench_reports/20260418-225920/report.json`
  - Resumen MD: `/tmp/rnfe_bench_reports/20260418-225920/report.md`
- Fase C:
  - Output JSON: `/tmp/rnfe_bench_reports/20260418-225836/phase_c.stdout.json`
  - Reporte: `/tmp/rnfe_bench_reports/20260418-230734/report.json`
  - Resumen MD: `/tmp/rnfe_bench_reports/20260418-230734/report.md`
  - Progreso/tiempo: `/tmp/rnfe_bench_reports/20260418-225836/phase_c.stderr.log`

## Fase A (Re-validación rápida)

**Comando ejecutado**:
```bash
pytest -q tests/reasoning_stress/test_pairwise_interaction.py \
  tests/reasoning_stress/test_hypercube_sampling.py \
  tests/reasoning_stress/test_adversarial_thresholds.py \
  tests/reasoning_stress/test_temporal_hysteresis.py \
  tests/reasoning_stress/test_family_contribution.py \
  tests/reasoning_stress/test_atlas_comprehensive.py \
  --tb=short --junitxml /tmp/rnfe_bench_reports/20260418-225836/phase_a/phase_a.xml
```

**Resultado**:
- Tests: `71`
- Passed: `71`
- Failed: `0`
- Errors: `0`
- Exit: `0`

## Fase B (W3 completa)

**Comando ejecutado**:
```bash
python scripts/benchmark_fractal_vs_nonfractal.py \
  --mode w3 \
  --base-dir /home/necktral/rnfe.v15 \
  --fractal-dir /tmp/rnfe-fractal-integrate-6ab844e \
  --out-root /tmp/rnfe_bench_reports
```

**Resultado de cierre**:
- `status=PASS`
- `ok_runs=5/5`

**Métricas W3 (fractal, media)**:
- `iters_per_sec`: `5208.594`
- `p50_us`: `181.239`
- `p95_us`: `277.021`
- `rss_kb`: `44983.2`

## Fase C (full)

**Comando ejecutado**:
```bash
python scripts/benchmark_fractal_vs_nonfractal.py \
  --mode full \
  --base-dir /home/necktral/rnfe.v15 \
  --fractal-dir /tmp/rnfe-fractal-integrate-6ab844e \
  --out-root /tmp/rnfe_bench_reports
```

**Tiempo total**:
- `21m34.48s` (wall-clock)

**Resultado global del runner**:
- `verdict = No eficiente`
- Totales: `tests=305`, `passed=242`, `failed=62`, `errors=1`

### Resultado por suites (bloqueantes vs informativas)

- Baseline common:
  - `common_regression`: `8/8` pass
  - `common_stress`: `84/84` pass
- Fractal common:
  - `common_regression`: `8/8` pass
  - `common_stress`: `70/85` pass (`14 fail`, `1 error`) **[bloqueante]**
- Fractal only (informativa en política, pero ejecutada completa):
  - `72/120` pass (`48 fail`)

### Primer fallo/traza relevante de Fase C

- Primer error bloqueante en fractal common:
  - `ERROR at setup of test_interaction_grid`
  - Archivo: `tests/reasoning_stress/test_pairwise_interaction.py`
- Primer síntoma funcional recurrente:
  - cancelación de familias críticas (`cau`, `ctf`) y falsas activaciones bajo umbral.

### Métricas de eficiencia (Fase C)

**Benchmark summaries (medias)**
- Baseline:
  - `W1`: `iters=48792.321`, `p95_us=25.421`, `rss_kb=17072`
  - `W2`: `iters=5394.608`, `p95_us=249.840`, `rss_kb=16896`
- Fractal:
  - `W1`: `iters=88862.319`, `p95_us=13.976`, `rss_kb=17024`
  - `W2`: `iters=5844.356`, `p95_us=220.447`, `rss_kb=17715.2`
  - `W3`: `iters=5257.411`, `p95_us=285.722`, `rss_kb=44981.6`

**Deltas reportados** (`analysis.deltas_pct`):
- `W1`: throughput `+82.12%`, p95 `-45.02%`, RSS `-0.28%`
- `W2`: throughput `+8.34%`, p95 `-11.76%`, RSS `+4.85%`

**Razón del veredicto `No eficiente`**:
- Aunque latencia/throughput no muestran degradación, hay regresión funcional severa en suites comunes fractales:
  - `common_pass_rate.drop_pp = 16.129`
  - `severe_functional_regression = true`

## Conclusión

- Fase A: cerrada en verde.
- Fase B: cerrada en verde (W3 5/5).
- Fase C: ejecutada de extremo a extremo, pero **no cumple gate** por regresión funcional en `fractal/common_stress` (bloqueante), y fractal_only también presenta fallos significativos.
- Estado final del cierre A/B/C: **ejecutado con hallazgos; baseline estable, fractal no apto para cierre Comparable en esta corrida**.

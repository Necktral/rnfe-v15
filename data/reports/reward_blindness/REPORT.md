# Ceguera de la recompensa: ¿un reward de coherencia suprime la efectividad?

## Afirmación

> Una recompensa de coherencia/proceso que NO mide efectividad suprime sistemáticamente capacidades que mejoran el resultado sin mejorar la coherencia.

Relevante a process/intrinsic rewards (process reward models, RLHF, motivación intrínseca): un proxy plausible de "buen razonamiento" (IoC) puede degradar el desempeño al eliminar comportamientos efectivos-pero-coherencia-neutros.

## Hipótesis (pre-registradas, fijadas antes de correr)

- **H1 (dosis-respuesta):** en tareas de conflicto, la tasa de retención/activación de la familia
  efectiva y la efectividad del mundo crecen monótonamente con λV; en λV=0 la familia queda suprimida.
- **H2 (especificidad / interacción):** el efecto de λV existe en conflicto y es NULO en tareas
  saturadas (sin brecha de efectividad que premiar).
- **H3 (control de ruido, sintético):** aleatorizar la señal de efectividad abole la recuperación ⇒
  el efecto es específico de la efectividad, no de "cualquier término extra en la recompensa".
- **Falsación:** retención alta con λV=0, o recuperación bajo el término de ruido, refutan la afirmación.

## Experimento A — Mecanismo (sintético, aísla la variable)

Conduce el selector guiado-por-recompensa con recompensas generadas donde coherencia es plana y efectividad depende de la familia efectiva (OPT). N=1000 semillas × 30 episodios por celda.

| λV | Retención (real) | Retención (ruido) |
|---:|---:|---:|
| 0.0 | 0.000 | 0.000 |
| 0.1 | 0.000 | 0.036 |
| 0.25 | 0.000 | 0.026 |
| 0.5 | 1.000 | 0.037 |
| 1.0 | 1.000 | 0.074 |

- **H1 dosis-respuesta**: ✓ (retención real [0.0, 0.0, 0.0, 1.0, 1.0]).
- **H1 suprimida en λV=0**: ✓.
- **H3 control de ruido plano**: ✓ (retención ruido [0.0, 0.036, 0.026, 0.037, 0.074]).

## Experimento B — Sistema real (confirma en la arquitectura)

Organismo completo (core_plus_deliberative ⇒ el selector gobierna plan/opt; actuación ON; conflicto RESETEADO cada episodio ⇒ estacionario), λV ∈ [0.0, 5.0, 20.0, 50.0], 8 semillas × 36 episodios. Tareas conflicto: ['t88_c4', 't88_c7']; saturadas: ['t80_c4', 't80_c7']. CI **entre-(tarea,semilla)**.

| Tarea | λV | Activación OPT [CI] | Efectividad media [CI] | Override rate |
|---|---:|---|---|---:|
| conflict | 0.0 | 0.000 [0.000,0.000] | -0.0257 [-0.0265,-0.0248] | 0.111 |
| conflict | 5.0 | 0.500 [0.250,0.750] | +0.0083 [+0.0017,+0.0150] | 0.736 |
| conflict | 20.0 | 1.000 [1.000,1.000] | +0.0099 [+0.0041,+0.0158] | 0.778 |
| conflict | 50.0 | 1.000 [1.000,1.000] | +0.0099 [+0.0041,+0.0158] | 0.778 |
| saturated | 0.0 | 0.000 [0.000,0.000] | +0.0543 [+0.0535,+0.0552] | 0.111 |
| saturated | 5.0 | 0.500 [0.250,0.750] | +0.0883 [+0.0817,+0.0950] | 0.736 |
| saturated | 20.0 | 1.000 [1.000,1.000] | +0.0899 [+0.0841,+0.0958] | 0.778 |
| saturated | 50.0 | 1.000 [1.000,1.000] | +0.0899 [+0.0841,+0.0958] | 0.778 |

- **H1 dosis-respuesta (conflicto)**: ✓ (activación [0.0, 0.5, 1.0, 1.0]).
- **H1 activación sube con λV (conflicto, λ0→λT)**: ✓ (Δ=+1.000, CI [+1.000,+1.000], d=0.0).
- Efectividad sube con λV (conflicto): Δ=+0.0356, CI [+0.0300,+0.0416], d=4.136.
- **H2 especificidad (saturado nulo)**: ✗ (Δ activación saturado=+1.000, CI [+1.000,+1.000] — EXCLUYE 0).

## Veredicto (por hipótesis, sin adornos)

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| **H1 — dosis-respuesta** | ✅ **CONFIRMADA** | Mecanismo: retención [0,0,0,1,1] en λV {0,.1,.25,.5,1}. Sistema real: activación OPT [0.0→0.5→1.0→1.0] en λV {0,5,20,50}; efectividad sube con λV (d=4.14, CI excluye 0). |
| **H3 — control de ruido (sintético)** | ✅ **CONFIRMADA** | Aleatorizar la señal de efectividad NO recupera la familia (retención ruido 0.0–0.074 frente a 1.0 con señal real) ⇒ el efecto es específico de la efectividad, no de "cualquier término extra en la recompensa". |
| **H2 — especificidad / interacción** | ❌ **REFUTADA** (honesto) | En tareas "saturadas" la activación también sube 0→1 con λV (Δ=+1.000, CI excluye 0). El térmico-*minimize* no ofrece un control sin-brecha (ver "Amenaza a la validez"). No contradice H1/H3; **acota** su alcance. |

**Tesis central (forma fuerte de la ceguera).** El reward de coherencia **no solo ignora** la
efectividad: **la penaliza**. El proxy IoC *anti-correlaciona* con la acción efectiva — ejecutar la
intervención desviada baja IoC de 0.888 a ~0.646 (−0.24), y la caída la domina el término de
**continuidad** (no Ω). Por eso la familia efectiva se suprime aun cuando mejora el mundo: la
coherencia-como-continuidad premia el conservadurismo. Recuperarla exige un término de efectividad
λV grande (ver abajo).

## Hallazgo del sistema real: el umbral salta ~40× (la idealización se rompe)

El mecanismo idealizado asume **coherencia plana** entre acciones; ahí el término de efectividad recupera la familia con λV≈0.5. En la arquitectura real eso NO se cumple: el proxy IoC **anti-correlaciona** con la acción efectiva — ejecutar la intervención desviada (override) baja IoC ~0.24 (de 0.888 a ~0.646), porque cambiar la conducta altera la estructura causal/cierre que IoC premia. Además ΔIoC* es un *delta* sobre una secuencia no-estacionaria ⇒ oscila ±0.24, ~70× la señal de efectividad a λV=0.5. Consecuencia: la recuperación de la familia efectiva exige λV≈**20** (no 0.5) — la señal de efectividad debe escalarse ~40× para superar el ruido/penalización de coherencia. Esto es una forma MÁS FUERTE de la ceguera: el reward de coherencia no solo ignora la efectividad, **penaliza activamente** la desviación efectiva (coherencia-como-continuidad premia el conservadurismo).

> **λV≈20 es un resultado CONTEXTUAL, no una constante universal.** El número depende del balance
> coste/efectividad y de la magnitud de la penalización de continuidad *en esta tarea térmica*; no es
> transferible a otras tareas ni a otras recompensas. El resultado científico no es el valor 20, sino
> **la existencia y la dirección** del fenómeno: el umbral real está muy por encima (~40×) del que
> predice el mecanismo idealizado de coherencia-plana, porque la coherencia real penaliza la desviación.

## Amenaza a la validez — H2 REFUTADA (honesto): no hay control 'sin-brecha' en el térmico

H2 predecía que λV NO activaría la familia efectiva en tareas saturadas (sin brecha de efectividad). Se **refuta**: en saturado la activación también sube 0→1 con λV. La causa es honesta — las tareas 'saturadas' (greedy no falla) NO son de verdad sin-brecha: el override MEJORA la efectividad incluso ahí (p.ej. +0.054→+0.088), porque la dirección de optimización es *minimize* y enfriar más SIEMPRE ayuda ⇒ siempre queda margen que la efectividad premia. Un control limpio de especificidad exige una tarea donde el greedy ya sea óptimo (dirección *target_band*, o greedy = acción efectiva), que el térmico-minimize no ofrece. Esto NO contradice H1/H3 (el efecto central existe y es graduado); acota su alcance: no pudimos demostrar que sea EXCLUSIVO de tareas con brecha, porque **no hay tarea sin brecha en este escenario**. Es la principal amenaza a la validez del estudio y la razón de que H2 quede como negativo honesto. Trabajo futuro: escenario con óptimo interior (*target_band*) que sí ofrezca el control limpio de especificidad.

## Limitaciones honestas

- La afirmación es sobre **especificación de la recompensa**, NO sobre sofisticación del razonamiento. La tarea es deliberadamente simple (térmica binaria) para AISLAR el efecto del reward de la dificultad de la tarea — no es evidencia de razonamiento avanzado.
- Una sola familia de tareas (térmica) y una señal de efectividad; la generalización a tareas ricas (observabilidad parcial, horizontes largos, ambigüedad) es trabajo futuro.
- λV no está calibrado contra un objetivo externo; el umbral de retención depende del balance coste/efectividad de esta tarea.
- Sin baseline contra métodos estándar (lookahead/RL): el override ES one-step lookahead; el estudio mide la SUPRESIÓN por la recompensa, no la potencia del razonamiento.

## Reproducibilidad

- **Commit**: `1fb81b9e` (`feat(study): experimento controlado de ceguera de la recompensa`).
- **Suite verificada**: `python -m pytest -q` ⇒ **946 passed, 0 failed** (13 skipped, 32 xfailed,
  1 xpassed; exit 0). Los 32 xfail son caracterización fractal conocida; sin fallos reales.
- **Sin cambios de runtime**: el estudio NO modifica la recompensa nominal. Solo barre dos flags de
  entorno —`RNFE_REWARD_LAMBDA_EFFECTIVENESS` (λV) y `RNFE_REASONING_ACTUATES`— y **lee** eventos
  `reasoning.reward`. Con los flags en su valor por defecto (off), el camino nominal es byte-idéntico.
- **Reproducir**: `python scripts/reward_blindness_study.py --mode both` (mecanismo en segundos;
  sistema real configurable). Las tablas de este informe provienen de `results.json`; el informe no
  re-corre el experimento.

## Disponibilidad de datos / higiene de disco

- El harness creaba un sqlite + artefactos por celda sin limpiarlos, llenando el `ext4.vhdx`. El leak
  fue **resuelto del lado Linux**: cada celda usa ahora un *scratch* efímero (tempdir + `rmtree`),
  verificado en 0 basura. `data/` pasó de **42G a 929M**.
- Los datos crudos completos (corridas previas de ganancia cognitiva y ecología) quedan
  **preservados en `D:\rnfe_data\`**: `cognitive_gain.tar.gz` (~193M) y `ecology.tar.gz` (~105M),
  verificados en `/mnt/d/rnfe_data/`.
- El repositorio conserva los entregables legibles de este estudio: este `REPORT.md` y `results.json`
  (de donde provienen todas las tablas; el informe no re-corre el experimento).

## Implicación futura (propuesta de diseño, NO implementada)

El estudio motiva — pero **no implementa aquí** — descomponer el escalar único IoC en **tres canales
ortogonales**. Hoy ([ioc_proxy.py:16-27](../../../runtime/certification/ioc_proxy.py)) IoC es una suma
ponderada `IoC = 0.45·continuity + 0.25·closure + 0.20·trace − 0.06·uncertainty − 0.14·collapse`. El
peso dominante de `continuity` es **exactamente** por qué la desviación efectiva se penaliza: cambiar
de conducta rompe la continuidad y arrastra el escalar a la baja, aunque el mundo mejore.

| Canal propuesto | Componentes hoy | Qué mediría | Por qué importa |
|---|---|---|---|
| **Coherencia estructural** | `closure`, `trace` | ¿Cierra el grafo de razonamiento? ¿La traza está intacta? | Proceso bien formado, independiente de la acción. |
| **Coherencia causal** | *(ausente hoy)* | ¿La acción elegida alinea con el modelo causal y mueve hacia el objetivo? | **El canal que falta**: dejaría que la efectividad registre *como coherencia*, en vez de como un término λV ajeno que hay que escalar ~40× para vencer la penalización de continuidad. |
| **Identidad** | `continuity` | Continuidad del auto-modelo entre episodios | Hoy domina (0.45) y **confunde "no cambiar" con "razonar bien"**; separarlo evita que la continuidad castigue la desviación efectiva. |

Con los tres canales separados, una desviación efectiva podría **subir** la coherencia causal mientras
baja la identidad — y un agregador podría premiar el resultado sin exigir un λV gigante. Es una
hipótesis de diseño que este negativo (la penalización de continuidad) hace falsable; encaja con el
roadmap canon de términos Ω / R-series. **No se toca en este cierre.**

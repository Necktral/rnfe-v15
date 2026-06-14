# Reward Coherence Blindness Study — commit 1fb81b9e

> Date: 2026-06-14 · Branch: `report/reward-blindness-1fb81b9e` · Study commit: `1fb81b9e`
> Scope: **documentation only**. No runtime, test, module, run-config or heavy-data changes.
> Companion artifacts (same folder): `REPORT.md` (Spanish, consolidated), `results.json` (raw),
> `manifest_1fb81b9e.yaml` (evidence manifest). Decision record: `docs/adr/ADR_REWARD_COHERENCE_BLINDNESS.md`.

## 1. Abstract

This study identifies a **measurable failure mode** in intrinsic / process rewards: a coherence
reward that scores *whether reasoning closed consistently* — but does **not** measure *whether the
action achieved the goal* — can **penalize** capabilities that improve the outcome without improving
coherence. In the RNFE architecture the coherence proxy IoC is dominated by a **continuity** term, so
executing an effective-but-deviating intervention **lowers** IoC even when it makes the world safer.
The result is not a general improvement and is not a claim about reasoning sophistication: it is a
negative result about **reward specification** — coherence-as-continuity can suppress causally
effective families unless an explicit effectiveness term is added, and the term must be scaled far
beyond what an idealized model predicts.

## 2. Scientific Question

Can a coherence reward **suppress** an effective reasoning family, and can an explicit effectiveness
term `λV` **recover** it?

Concretely, in a conflict task where the greedy/coherent action is wrong and a deviating action is
correct: (a) does a coherence-only reward drive the selector to drop the effective family, and (b)
does adding `r += λV · effectiveness` restore it, and at what magnitude of `λV`?

## 3. Hypotheses (pre-registered, fixed before running)

- **H1 — dose-response: CONFIRMED.** Activation/retention of the effective family and world
  effectiveness rise monotonically with `λV`; at `λV=0` the family is suppressed.
- **H2 — specificity / interaction: REFUTED (honestly).** The `λV` effect was expected to be present
  in conflict tasks and **absent** in saturated (no-gap) tasks. It was **not** absent — see §7.
  H2 is refuted because the **thermal control did not offer a no-gap condition**: in a *minimize*
  direction, cooling further always helps, so the "saturated" tasks still had an effectiveness gap
  the term could reward. The refutation bounds the claim's scope; it does **not** contradict H1/H3.
- **H3 — noise control (synthetic): CONFIRMED.** Randomizing (decorrelating) the effectiveness signal
  from the family abolishes recovery ⇒ the effect is specific to the **effectiveness signal**, not to
  "any extra term in the reward".

We distinguish three verdict states throughout: **confirmed**, **refuted**, and **not isolated**
(H2's specificity could not be isolated in this scenario).

## 4. Experimental Regimes

| Regime | What it isolates | Setup |
|---|---|---|
| **Idealized mechanism** (synthetic) | The variable alone, at high N | Drives `RewardGuidedOverlaySelector` directly with synthetic rewards where **coherence is flat** and **effectiveness = +gap iff the effective family is active**. 1000 seeds × 30 episodes per cell; `λV ∈ {0, 0.1, 0.25, 0.5, 1.0}`. |
| **Real system** (ecological) | Confirmation inside the architecture | Full organism (`core_plus_deliberative` so the selector governs plan/opt; actuation ON; conflict **reset each episode** ⇒ stationary). `λV ∈ {0, 5, 20, 50}`, 8 seeds × 36 episodes; between-seed bootstrap CI. Reads `reasoning.reward` events; **no runtime change**. |
| **Noise control** (synthetic) | Whether *any* extra term recovers the family | Same mechanism with the effectiveness signal **shuffled** (decorrelated). If recovery persists, the claim is false. |

The mechanism vs real-system split is deliberately preserved: the gap between them is itself a result
(§5).

## 5. Main Results

- **`λV` real threshold ≈ 20.** In the real system, OPT activation goes `0.0 → 0.5 → 1.0 → 1.0`
  across `λV ∈ {0, 5, 20, 50}`; full recovery requires `λV ≈ 20`.
- **Real threshold ≈ 40× the idealized one.** The idealized mechanism (flat coherence) recovers the
  family at `λV ≈ 0.5`; the real system needs `≈ 20` — roughly **40×** higher.
- **IoC anti-correlates with the effective override.** Executing the deviating (correct) intervention
  **lowers** IoC from `0.888 → 0.646` (−0.24). The drop is dominated by the **continuity** term, not
  by Ω. `ΔIoC*` is a noisy delta over a sequence (±0.24, ~70× the effectiveness signal at `λV=0.5`).
- **The coherence reward penalizes effective deviation.** Suppression is not mere indifference: the
  effective family scores *worse* on coherence precisely because deviating breaks continuity. This is
  the stronger form of the blindness.
- **Effectiveness rises with `λV` in conflict** (Cohen `d ≈ 4.14`, between-seed CI excludes 0).
- **Synthetic dose-response** retention `[0,0,0,1,1]`; **noise control** retention `0.0–0.074`
  (flat) ⇒ H3 holds.
- **Suite: 946 passed, 0 failed** (13 skipped, 32 xfailed, 1 xpassed; exit 0). The 32 xfail are known
  fractal-characterization cases, not real failures.
- **No runtime changes.** The study only sweeps two env flags (`RNFE_REWARD_LAMBDA_EFFECTIVENESS`,
  `RNFE_REASONING_ACTUATES`) and **reads** events; with flags at their default (off) the nominal path
  is byte-identical.

## 6. Interpretation

This is **not** a general improvement, **not** a gain, and **not** evidence of advanced reasoning. It
is a **measured failure mode** of process/intrinsic reward specification: a plausible "good reasoning"
proxy (closure integrity) can actively degrade outcomes by removing effective-but-coherence-neutral
(indeed coherence-*negative*) behaviors.

> **Coherence-as-continuity can penalize effective deviation.**

The practical reading: when a process reward rewards continuity/closure, the agent is incentivized
toward conservatism, and an effectiveness term must be added — and **scaled well beyond** the naive
estimate — to merely break even against the continuity penalty.

## 7. Validity Threats

- **`λV ≈ 20` is not a universal constant.** It is a **contextual** result of this task's
  cost/effectiveness balance and the magnitude of the continuity penalty; it is not transferable to
  other tasks or other rewards. The scientific content is the **existence and direction** of the
  penalty (threshold ≫ idealized), not the number.
- **H2 is refuted**, not pending: specificity could not be demonstrated.
- **The thermal control did not isolate a no-gap condition.** "Saturated" tasks still had cooling
  headroom (minimize direction), so the effectiveness term had something to reward there too.
- **Possible metric coupling.** IoC bundles continuity, closure and trace into one scalar; the
  observed anti-correlation may partly reflect that coupling rather than a clean coherence/effect
  trade-off.
- **Possible non-stationarity.** The conflict self-resolves once cooling is applied; the real-system
  cells reset the scenario each episode to enforce stationarity, but residual dynamics may remain.
- **Risk of over-interpreting IoC as "universal coherence."** IoC here is a specific operational
  proxy, not a general definition of coherence; conclusions are about *this* proxy.
- **Dataset / config specificity.** A single task family (thermal-binary), one effectiveness signal,
  one selector configuration. Generalization to rich tasks (partial observability, long horizons,
  ambiguity) is future work.

## 8. RNFE Architectural Implication (proposed, not implemented)

The study **motivates — but does not implement** — decomposing the single IoC scalar
(`runtime/certification/ioc_proxy.py`, currently
`IoC = 0.45·continuity + 0.25·closure + 0.20·trace − 0.06·uncertainty − 0.14·collapse`) into three
orthogonal channels:

- **`IoC_struct`** (structural coherence) — `closure`, `trace`: is the reasoning graph well-formed and
  the trace intact, independent of the action chosen.
- **`IoC_causal`** (causal coherence) — *absent today*: does the chosen action align with the causal
  model and move toward the goal. This is the missing channel that would let effectiveness register
  **as coherence** instead of as a foreign `λV` term scaled ~40× to overcome continuity.
- **`IoC_identity`** (identity) — `continuity`: continuity of the self-model across episodes; today it
  dominates (weight 0.45) and conflates "do not change" with "reason well".

With the channels separated, an effective deviation could **raise** `IoC_causal` while lowering
`IoC_identity`, and an aggregator could reward the outcome without a giant `λV`. This is a design
hypothesis the negative result makes falsable. **It is not implemented in this study or PR.**

## 9. Reproducibility

- **Commit:** `1fb81b9e` (study code; this report is the documentation close-out).
- **Suite:** 946 passed, 0 failed (13 skipped, 32 xfailed, 1 xpassed; `python -m pytest -q`, exit 0).
- **`runtime_changes`: false** — env-only sweep + event reads; nominal reward unchanged.
- **Full data preserved in `D:\rnfe_data\`**: `cognitive_gain.tar.gz` (~193M), `ecology.tar.gz`
  (~105M), verified at `/mnt/d/rnfe_data/`.
- **Repo contains readable reports**: this file, `REPORT.md`, `results.json` (source of all tables).
- **Linux data leak fixed**: per-cell sqlite + artifacts are now written to an ephemeral scratch
  (tempdir + `rmtree`); `data/` went from **42G → 929M**.
- **Windows C: note**: freed space is not returned to C: until the WSL VHDX is compacted
  (`wsl --shutdown` + `wsl --manage <distro> --set-sparse true`); `fstrim` returns 0 because the
  vhdx is not sparse. This is a user-side step, only if reclaiming C: is needed.
- **Reproduce:** `python scripts/reward_blindness_study.py --mode both` (mechanism in seconds, real
  system configurable). The report does not re-run the experiment.

## 10. Conclusion

The study confirms a measurable coherence-blindness mechanism and rejects unsupported specificity
claims. H1 and H3 are confirmed; H2 is refuted honestly because the thermal scenario offered no
no-gap control. `λV ≈ 20` is reported as a contextual threshold, not a universal law. The deliverable
is a measured failure mode plus an honest negative — not a gain, multiplication, or general
improvement.

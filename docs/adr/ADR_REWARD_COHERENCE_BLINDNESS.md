# ADR: Reward Coherence Blindness — IoC penalizes effective deviation

- **Status:** proposed (non-normative finding; not a canon change)
- **Date:** 2026-06-14
- **Study commit:** 1fb81b9e
- **Scope of this ADR/PR:** documentation only — no runtime, tests, modules, run-config or heavy data.
- **Evidence:** `data/reports/reward_blindness/2026-06-14_coherence_reward_blindness.md`,
  `REPORT.md`, `results.json`, `manifest_1fb81b9e.yaml`.
- **Canon cross-reference:** `docs/analysis/AUDITORIA_CANON_F21_F24_VS_CODIGO.md` (IoC is a canonical
  quantity; this ADR proposes, it does not redefine canon).

## Context

The reward-blindness study (commit `1fb81b9e`) measured a failure mode in the process/coherence
reward: the IoC proxy (`runtime/certification/ioc_proxy.py`,
`IoC = 0.45·continuity + 0.25·closure + 0.20·trace − 0.06·uncertainty − 0.14·collapse`) is dominated
by its **continuity** term. In a conflict task where the greedy/coherent action is wrong, executing
the correct deviating intervention **lowers** IoC (`0.888 → 0.646`), so the reward-guided selector
suppresses the effective family. An explicit effectiveness term recovers it, but only at
`λV ≈ 20` — about **40×** the value an idealized flat-coherence model predicts (`≈ 0.5`).

Verdicts: **H1 confirmed**, **H3 confirmed**, **H2 refuted honestly** (the thermal-*minimize*
scenario offers no no-gap control, so specificity could not be isolated). `λV ≈ 20` is a
**contextual** result, not a universal constant.

Central finding, verbatim: **coherence-as-continuity can penalize effective deviation** — not merely
ignore it.

## Decision

1. **Record the failure mode** as a measured, falsifiable negative result. It is not a gain or a
   general improvement.
2. **Do NOT replace or modify IoC yet.** The current IoC proxy and the nominal reward path remain
   unchanged. The effectiveness term (`λV`) and the override stay **off by default** (shadow flags).
3. **No runtime changes in this PR.** This is a documentation close-out; the only artifacts are
   reports and this ADR.
4. **Open future work** (separate, runtime-touching PR, gated and tested) to **decouple coherence
   into three orthogonal channels**: structural coherence, causal coherence, and identity.

## Consequences

- The repository carries a reviewable, self-contained record of the failure mode without changing
  behavior; the nominal path is byte-identical with default flags.
- Any future use of an intrinsic/coherence reward in RNFE is on notice that continuity-dominated
  coherence can penalize effective deviation, and that an effectiveness term may need to be scaled
  far above naive estimates to compensate.
- `λV ≈ 20` must **not** be treated as a tunable universal constant; it is task-contextual.
- Because IoC bundles continuity/closure/trace into a single scalar, the observed anti-correlation may
  reflect metric coupling; this is acknowledged as a validity threat, not a proven clean trade-off.

## Non-goals

- **Not** replacing or reweighting IoC in this PR.
- **Not** enabling the effectiveness term or the override in the nominal runtime.
- **Not** claiming reasoning sophistication, cognitive gain, multiplication, or general improvement.
- **Not** asserting that `λV ≈ 20` generalizes beyond the thermal task.
- **Not** redefining the canonical IoC (f2.1–f2.4); this ADR is non-normative and proposes only.

## Future work

Decouple the single IoC scalar into three orthogonal channels (proposed, **not implemented**):

- **`IoC_struct`** — structural coherence (`closure`, `trace`): is the reasoning graph well-formed and
  the trace intact, independent of the action.
- **`IoC_causal`** — causal coherence (*absent today*): does the chosen action align with the causal
  model and move toward the goal. This is the missing channel that would let effectiveness register
  *as coherence* instead of as a foreign `λV` term scaled ~40× to overcome continuity.
- **`IoC_identity`** — identity (`continuity`): continuity of the self-model; today it dominates
  (0.45) and conflates "do not change" with "reason well".

A future PR would implement these channels behind shadow flags, with tests, a no-gap control task
(e.g. an interior-optimum / `target_band` scenario to finally isolate H2's specificity), and a
calibration of any aggregator. This ADR only authorizes opening that work, not performing it.

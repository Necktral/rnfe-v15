# Reward Coherence Blindness — study folder

Closed study (commit `1fb81b9e`) on a **measured failure mode** of process/coherence rewards:
**coherence-as-continuity can penalize effective deviation.** Not a gain, not a general improvement —
a falsifiable negative result about reward specification.

## Contents

| File | What it is |
|---|---|
| [`2026-06-14_coherence_reward_blindness.md`](2026-06-14_coherence_reward_blindness.md) | **Main report** (English, 10 sections: abstract → conclusion). Start here. |
| [`REPORT.md`](REPORT.md) | Consolidated report (Spanish), with the full result tables and per-hypothesis verdict. |
| [`results.json`](results.json) | Raw results — source of every table. The reports do not re-run the experiment. |
| [`manifest_1fb81b9e.yaml`](manifest_1fb81b9e.yaml) | Evidence manifest: commit, verdicts, suite result, claim, limitations, data location. |
| `_system/` | Local scratch (untracked). Not part of the deliverable. |

Architectural decision record:
[`docs/adr/ADR_REWARD_COHERENCE_BLINDNESS.md`](../../../docs/adr/ADR_REWARD_COHERENCE_BLINDNESS.md).

## Verdict at a glance

- **H1 (dose-response): confirmed** — effective family suppressed at `λV=0`, recovered as `λV` rises.
- **H3 (noise control): confirmed** — shuffling the effectiveness signal does **not** recover the family.
- **H2 (specificity): refuted honestly** — the thermal-*minimize* scenario offers no no-gap control,
  so specificity could **not be isolated**. Bounds the claim; does not contradict H1/H3.
- **`λV ≈ 20`** real threshold, ≈ **40×** the idealized one — reported as **contextual**, not a
  universal law.
- **Key finding:** IoC **anti-correlates** with the effective override (IoC `0.888 → 0.646`, driven by
  the continuity term) ⇒ the coherence reward **penalizes** effective deviation, not merely ignores it.

## Reproduce / verify

- Suite: `python -m pytest -q` ⇒ **946 passed, 0 failed**.
- Study: `python scripts/reward_blindness_study.py --mode both`.
- **No runtime changes**: env-only sweep (`RNFE_REWARD_LAMBDA_EFFECTIVENESS`,
  `RNFE_REASONING_ACTUATES`) + event reads.

## Data availability

Full raw runs preserved in `D:\rnfe_data\` (`cognitive_gain.tar.gz`, `ecology.tar.gz`). The Linux-side
per-cell leak is fixed (ephemeral scratch); `data/` went 42G → 929M.

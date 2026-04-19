from __future__ import annotations

import itertools
import math
import random
import statistics
from typing import Iterable


EPS = 1e-12


def _finite(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    for raw in values:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compose_weighted_score(
    components: dict[str, float],
    weights: dict[str, float],
    *,
    scale: float = 100.0,
    lo: float = 0.0,
    hi: float = 100.0,
    renormalize: bool = True,
) -> dict[str, float | dict[str, float]]:
    active = {k: float(v) for k, v in weights.items() if k in components}
    if not active:
        return {"score": float("nan"), "contributions": {}}
    total_weight = sum(abs(v) for v in active.values()) if renormalize else 1.0
    total_weight = total_weight if total_weight > EPS else 1.0
    contributions: dict[str, float] = {}
    score = 0.0
    for name, w in active.items():
        norm_w = (w / total_weight) if renormalize else w
        comp = clip(float(components[name]), 0.0, 1.0)
        c = scale * norm_w * comp
        contributions[name] = c
        score += c
    return {"score": clip(score, lo, hi), "contributions": contributions}


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


def bootstrap_ci_delta_abs(
    base_vals: Iterable[float],
    other_vals: Iterable[float],
    *,
    n_boot: int,
    seed: int = 1337,
) -> dict[str, float | bool | int]:
    base = _finite(base_vals)
    other = _finite(other_vals)
    if not base or not other:
        return {
            "point": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci95_degenerate": True,
            "n_base": len(base),
            "n_other": len(other),
            "base_var": float("nan"),
            "other_var": float("nan"),
        }

    rng = random.Random(seed)
    n0 = len(base)
    n1 = len(other)
    point = statistics.mean(other) - statistics.mean(base)
    boots: list[float] = []
    for _ in range(max(1, n_boot)):
        s0 = [base[rng.randrange(n0)] for _ in range(n0)]
        s1 = [other[rng.randrange(n1)] for _ in range(n1)]
        boots.append(statistics.mean(s1) - statistics.mean(s0))
    boots.sort()
    low = _percentile(boots, 0.025)
    high = _percentile(boots, 0.975)
    return {
        "point": point,
        "ci95_low": low,
        "ci95_high": high,
        "ci95_degenerate": abs(high - low) <= EPS,
        "n_base": n0,
        "n_other": n1,
        "base_var": statistics.pvariance(base) if n0 >= 1 else float("nan"),
        "other_var": statistics.pvariance(other) if n1 >= 1 else float("nan"),
    }


def bootstrap_ci_delta_pct(
    base_vals: Iterable[float],
    other_vals: Iterable[float],
    *,
    n_boot: int,
    seed: int = 1337,
) -> dict[str, float | bool | int]:
    base = _finite(base_vals)
    other = _finite(other_vals)
    if not base or not other:
        return {
            "point_pct": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci95_degenerate": True,
            "n_base": len(base),
            "n_other": len(other),
            "base_var": float("nan"),
            "other_var": float("nan"),
        }

    base_mean = statistics.mean(base)
    if abs(base_mean) <= EPS:
        return {
            "point_pct": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci95_degenerate": True,
            "n_base": len(base),
            "n_other": len(other),
            "base_var": statistics.pvariance(base),
            "other_var": statistics.pvariance(other),
        }

    rng = random.Random(seed)
    n0 = len(base)
    n1 = len(other)
    point = ((statistics.mean(other) - base_mean) / base_mean) * 100.0
    boots: list[float] = []
    for _ in range(max(1, n_boot)):
        s0 = [base[rng.randrange(n0)] for _ in range(n0)]
        s1 = [other[rng.randrange(n1)] for _ in range(n1)]
        m0 = statistics.mean(s0)
        if abs(m0) <= EPS:
            continue
        m1 = statistics.mean(s1)
        boots.append(((m1 - m0) / m0) * 100.0)
    boots.sort()
    low = _percentile(boots, 0.025)
    high = _percentile(boots, 0.975)
    return {
        "point_pct": point,
        "ci95_low": low,
        "ci95_high": high,
        "ci95_degenerate": abs(high - low) <= EPS if boots else True,
        "n_base": n0,
        "n_other": n1,
        "base_var": statistics.pvariance(base),
        "other_var": statistics.pvariance(other),
    }


def hedges_g(base_vals: Iterable[float], other_vals: Iterable[float]) -> dict[str, float | bool | str]:
    base = _finite(base_vals)
    other = _finite(other_vals)
    n0 = len(base)
    n1 = len(other)
    if n0 < 2 or n1 < 2:
        return {
            "value": float("nan"),
            "undefined": True,
            "reason": "insufficient_n",
            "n_base": n0,
            "n_other": n1,
        }

    v0 = statistics.variance(base)
    v1 = statistics.variance(other)
    pooled = (((n0 - 1) * v0) + ((n1 - 1) * v1)) / max(1, (n0 + n1 - 2))
    if pooled <= EPS:
        return {
            "value": float("nan"),
            "undefined": True,
            "reason": "zero_pooled_variance",
            "n_base": n0,
            "n_other": n1,
        }

    m0 = statistics.mean(base)
    m1 = statistics.mean(other)
    d = (m1 - m0) / math.sqrt(pooled)
    correction = 1.0 - (3.0 / max(1.0, (4.0 * (n0 + n1) - 9.0)))
    return {
        "value": d * correction,
        "undefined": False,
        "reason": "ok",
        "n_base": n0,
        "n_other": n1,
    }


def cliffs_delta(
    base_vals: Iterable[float],
    other_vals: Iterable[float],
    *,
    larger_is_better: bool = True,
) -> dict[str, float]:
    base = _finite(base_vals)
    other = _finite(other_vals)
    if not base or not other:
        return {"value": float("nan"), "n_base": len(base), "n_other": len(other)}
    gt = 0
    lt = 0
    for x in other:
        for y in base:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    total = len(other) * len(base)
    raw = (gt - lt) / total if total else float("nan")
    return {
        "value": raw if larger_is_better else -raw,
        "n_base": len(base),
        "n_other": len(other),
    }


def wilson_interval(success: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total <= 0:
        return {"center": 0.0, "low": 0.0, "high": 0.0}
    p = success / total
    denom = 1.0 + (z * z / total)
    center = (p + (z * z / (2.0 * total))) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))
    return {
        "center": center,
        "low": max(0.0, center - spread),
        "high": min(1.0, center + spread),
    }


def permutation_test_mean_delta(
    base_vals: Iterable[float],
    other_vals: Iterable[float],
    *,
    n_perm: int = 50000,
    seed: int = 1337,
) -> dict[str, float | int | str | bool]:
    base = _finite(base_vals)
    other = _finite(other_vals)
    n0 = len(base)
    n1 = len(other)
    if n0 == 0 or n1 == 0:
        return {
            "observed_delta": float("nan"),
            "p_two_sided": float("nan"),
            "method": "invalid",
            "n_base": n0,
            "n_other": n1,
            "num_samples": 0,
        }
    observed = statistics.mean(other) - statistics.mean(base)
    combined = base + other
    total_n = len(combined)
    exact_count = math.comb(total_n, n1)
    threshold = abs(observed) - EPS

    if exact_count <= n_perm:
        extreme = 0
        for idxs in itertools.combinations(range(total_n), n1):
            idx_set = set(idxs)
            grp1 = [combined[i] for i in range(total_n) if i in idx_set]
            grp0 = [combined[i] for i in range(total_n) if i not in idx_set]
            delta = statistics.mean(grp1) - statistics.mean(grp0)
            if abs(delta) >= threshold:
                extreme += 1
        p_val = extreme / max(1, exact_count)
        return {
            "observed_delta": observed,
            "p_two_sided": p_val,
            "method": "exact",
            "n_base": n0,
            "n_other": n1,
            "num_samples": exact_count,
        }

    rng = random.Random(seed)
    extreme = 0
    trials = max(1, n_perm)
    for _ in range(trials):
        perm = combined[:]
        rng.shuffle(perm)
        grp1 = perm[:n1]
        grp0 = perm[n1:]
        delta = statistics.mean(grp1) - statistics.mean(grp0)
        if abs(delta) >= threshold:
            extreme += 1
    p_val = (extreme + 1) / (trials + 1)
    return {
        "observed_delta": observed,
        "p_two_sided": p_val,
        "method": "monte_carlo",
        "n_base": n0,
        "n_other": n1,
        "num_samples": trials,
    }

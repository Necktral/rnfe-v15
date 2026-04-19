from __future__ import annotations

import math

from tests.reasoning_stress.ablation_stats import (
    bootstrap_ci_delta_abs,
    cliffs_delta,
    compose_weighted_score,
    hedges_g,
    permutation_test_mean_delta,
    wilson_interval,
)


def test_stats_identical_groups_are_neutral() -> None:
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    other = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = bootstrap_ci_delta_abs(base, other, n_boot=2000, seed=11)
    assert abs(ci["point"]) < 1e-9
    assert ci["ci95_low"] <= 0.0 <= ci["ci95_high"]

    g = hedges_g(base, other)
    assert not g["undefined"]
    assert abs(g["value"]) < 1e-9

    d = cliffs_delta(base, other)
    assert abs(d["value"]) < 1e-9

    perm = permutation_test_mean_delta(base, other, n_perm=50000, seed=7)
    assert perm["p_two_sided"] > 0.5


def test_stats_separated_groups_have_consistent_direction() -> None:
    base = [1.0, 1.5, 2.0, 2.5, 3.0]
    other = [8.0, 9.0, 10.0, 11.0, 12.0]
    ci = bootstrap_ci_delta_abs(base, other, n_boot=3000, seed=3)
    assert ci["point"] > 0.0
    assert ci["ci95_low"] > 0.0
    assert not ci["ci95_degenerate"]

    g = hedges_g(base, other)
    assert not g["undefined"]
    assert g["value"] > 1.0

    d = cliffs_delta(base, other)
    assert d["value"] > 0.8

    perm = permutation_test_mean_delta(base, other, n_perm=50000, seed=13)
    assert perm["p_two_sided"] < 0.05


def test_hedges_g_is_undefined_when_zero_variance() -> None:
    base = [2.0, 2.0, 2.0, 2.0]
    other = [5.0, 5.0, 5.0, 5.0]
    g = hedges_g(base, other)
    assert g["undefined"] is True
    assert g["reason"] == "zero_pooled_variance"
    assert math.isnan(g["value"])


def test_ci_not_degenerate_when_variation_exists() -> None:
    base = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    other = [2.0, 4.0, 3.0, 5.0, 4.0, 6.0]
    ci = bootstrap_ci_delta_abs(base, other, n_boot=2500, seed=19)
    assert ci["ci95_high"] > ci["ci95_low"]
    assert not ci["ci95_degenerate"]


def test_cliffs_delta_orientation_lower_is_better() -> None:
    base = [10.0, 11.0, 12.0]
    other = [5.0, 6.0, 7.0]
    d_default = cliffs_delta(base, other, larger_is_better=True)
    d_inverted = cliffs_delta(base, other, larger_is_better=False)
    assert d_default["value"] < 0.0
    assert d_inverted["value"] > 0.0


def test_wilson_interval_bounds() -> None:
    interval = wilson_interval(84, 84)
    assert 0.9 <= interval["low"] <= 1.0
    assert math.isclose(interval["high"], 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_compose_weighted_score_is_traceable() -> None:
    components = {"a": 0.5, "b": 1.0, "c": 0.0}
    weights = {"a": 0.5, "b": 0.5, "c": 0.1}
    out = compose_weighted_score(components, weights, scale=100.0, renormalize=True)
    assert 0.0 <= out["score"] <= 100.0
    contrib = out["contributions"]
    assert "a" in contrib and "b" in contrib and "c" in contrib
    assert contrib["b"] > contrib["a"] >= contrib["c"]

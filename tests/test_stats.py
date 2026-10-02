import pytest

from text2sql.eval.stats import wilson, wilson_pct


@pytest.mark.parametrize(
    "k, n, lo, hi",
    [
        # reference values: roots of (p - p0)^2 = z^2 p0 (1 - p0) / n, solved separately
        (50, 100, 0.403832, 0.596168),
        (0, 10, 0.0, 0.277533),
        (10, 10, 0.722467, 1.0),
        (1, 20, 0.008881, 0.236132),
        (239, 500, 0.434551, 0.521785),
    ],
)
def test_matches_reference_values(k, n, lo, hi):
    got = wilson(k, n)
    assert got[0] == pytest.approx(lo, abs=1e-5)
    assert got[1] == pytest.approx(hi, abs=1e-5)


def test_half_width_at_500_is_about_4_4_points():
    lo, hi = wilson_pct(250, 500)
    assert (hi - lo) / 2 == pytest.approx(4.37, abs=0.02)


def test_interval_contains_point_estimate_and_stays_in_bounds():
    for n in (1, 7, 50, 500):
        for k in range(0, n + 1, max(1, n // 7)):
            lo, hi = wilson(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_empty_sample_is_uninformative():
    assert wilson(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize("k, n", [(-1, 5), (6, 5), (0, -1)])
def test_bad_input(k, n):
    with pytest.raises(ValueError):
        wilson(k, n)


def test_agrees_with_direct_quadratic_solution():
    z = 1.959964
    for n in (3, 50, 500):
        for k in range(n + 1):
            p = k / n
            a = 1 + z * z / n
            b = -(2 * p + z * z / n)
            disc = (b * b - 4 * a * p * p) ** 0.5
            lo, hi = wilson(k, n)
            assert lo == pytest.approx(max(0.0, (-b - disc) / (2 * a)), abs=1e-9)
            assert hi == pytest.approx(min(1.0, (-b + disc) / (2 * a)), abs=1e-9)

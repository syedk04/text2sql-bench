"""Confidence intervals for accuracy numbers."""

from __future__ import annotations

import math

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n``, as proportions.

    Unlike the normal approximation it stays inside [0, 1] and behaves at 0/n
    and n/n, which matters for small slices like the 10 hard questions in the
    dev manifest.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_pct(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    lo, hi = wilson(k, n, z)
    return (100 * lo, 100 * hi)

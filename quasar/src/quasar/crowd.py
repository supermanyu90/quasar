"""Crowd fluid dynamics: the Weidmann speed-density relation and Fruin LOS.

Two forms of the speed-density curve are implemented.

``weidmann_speed`` is the exponential-in-density form mandated by the Quasar
specification:

    v(rho) = v_max * (1 - exp(-gamma * (rho_max - rho)))

``weidmann_canonical_speed`` is Weidmann's published form, which is exponential
in *reciprocal* density (i.e. in pedestrian spacing):

    v(rho) = v_max * (1 - exp(-gamma_c * (1/rho - 1/rho_max)))

The two are not equivalent, and the difference is safety-relevant. We fit the
mandated form's single free parameter by least squares against the canonical
curve over the operational band (``tools/calibrate_gamma.py``; ``test_crowd.py``
asserts the shipped constant is still the optimum). The best achievable fit
still leaves an RMSE of 0.21 m/s, and the residual is *signed*: the mandated
form under-predicts speed at free flow and **over-predicts it under congestion**
(at rho = 3.0 ped/m^2 it returns 0.54 m/s against Weidmann's 0.33 m/s). An
over-predicted speed in a jammed corridor is precisely the error a router must
not make -- it prices a dangerous edge as cheap.

So we retain the mandated form and strengthen it, as the specification asks:

* gamma is calibrated, not hand-picked (``GAMMA_MANDATED``);
* every speed that feeds a *decision* -- route cost, ETA, evacuation margin --
  goes through :func:`safe_speed`, the pointwise minimum of the two curves.
  The mandated form is therefore never allowed to flatter a congested edge,
  while remaining the model of record wherever it is the conservative one.

The residual is reported in the submission's Limitations section rather than
buried here.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

from quasar.types import LOS

# Weidmann (1993) free-flow walking speed and jam density for mixed adult crowds.
V_FREE_M_S: float = 1.34
RHO_JAM_PED_M2: float = 5.4

# Canonical Weidmann shape parameter (units: ped/m^2, applied to spacing).
GAMMA_CANONICAL: float = 1.913

# Shape parameter for the mandated form, fitted to GAMMA_CANONICAL over
# OPERATIONAL_BAND by tools/calibrate_gamma.py (RMSE 0.207 m/s). Units: m^2/ped.
GAMMA_MANDATED: float = 0.2144

# Band over which the fit is performed and over which the mandated form is
# trusted. Below 0.2 ped/m^2 pedestrians are effectively at free flow; above
# jam density the model is undefined.
OPERATIONAL_BAND: tuple[float, float] = (0.2, RHO_JAM_PED_M2)

# Numerical floor so route cost stays finite as rho -> rho_max. A pedestrian at
# jam density is not moving; the router must still be able to price the edge.
V_MIN_M_S: float = 0.05

def weidmann_speed(rho: float, *, gamma: float = GAMMA_MANDATED) -> float:
    """Mandated speed-density form. Returns walking speed in m/s.

    Raises ValueError on negative density; clamps at jam density.
    """
    if rho < 0.0:
        raise ValueError(f"density must be non-negative, got {rho}")
    if rho >= RHO_JAM_PED_M2:
        return V_MIN_M_S
    v = V_FREE_M_S * (1.0 - math.exp(-gamma * (RHO_JAM_PED_M2 - rho)))
    return max(v, V_MIN_M_S)


def weidmann_canonical_speed(rho: float, *, gamma: float = GAMMA_CANONICAL) -> float:
    """Weidmann's published spacing-exponential form. Returns m/s."""
    if rho < 0.0:
        raise ValueError(f"density must be non-negative, got {rho}")
    if rho <= 1e-9:
        return V_FREE_M_S
    if rho >= RHO_JAM_PED_M2:
        return V_MIN_M_S
    v = V_FREE_M_S * (1.0 - math.exp(-gamma * (1.0 / rho - 1.0 / RHO_JAM_PED_M2)))
    return max(v, V_MIN_M_S)


def safe_speed(rho: float) -> float:
    """Conservative envelope of the two models -- the speed of record for every
    decision (route cost, ETA, evacuation margin).

    Taking the pointwise minimum guarantees no edge is ever priced as faster
    than *both* models believe it to be. This is what stops the mandated form's
    congested-regime over-prediction from making a jammed corridor look cheap.
    """
    return min(weidmann_speed(rho), weidmann_canonical_speed(rho))


def specific_flow(rho: float, *, speed: Callable[[float], float] = safe_speed) -> float:
    """Specific flow J = rho * v(rho), in ped / (m . s)."""
    return rho * speed(rho)


def corridor_capacity(rho: float, width_m: float) -> float:
    """Throughput of a corridor of the given width at the given density, ped/s."""
    if width_m <= 0.0:
        raise ValueError("width must be positive")
    return specific_flow(rho) * width_m


# Fruin walkway LOS, expressed as upper density bounds (ped/m^2), obtained by
# inverting Fruin's pedestrian-area module thresholds.
_LOS_UPPER_BOUNDS: tuple[tuple[float, LOS], ...] = (
    (0.308, LOS.A),
    (0.431, LOS.B),
    (0.719, LOS.C),
    (1.075, LOS.D),
    (2.174, LOS.E),
    (math.inf, LOS.F),
)

# Operational triggers used by the deterministic plane.
ADVISORY_DENSITY: float = 1.075  # LOS D/E boundary -- start shaping flow
CRITICAL_DENSITY: float = 2.174  # LOS E/F boundary -- mandatory reroute + cordon


def level_of_service(rho: float) -> LOS:
    """Fruin LOS letter for a walkway at the given density."""
    if rho < 0.0:
        raise ValueError(f"density must be non-negative, got {rho}")
    for upper, los in _LOS_UPPER_BOUNDS:
        if rho < upper:
            return los
    return LOS.F  # unreachable: last bound is inf


def is_critical(rho: float) -> bool:
    return rho >= CRITICAL_DENSITY


def is_advisory(rho: float) -> bool:
    return rho >= ADVISORY_DENSITY


def fit_gamma(
    *,
    candidates: Iterable[float] | None = None,
    band: tuple[float, float] = OPERATIONAL_BAND,
    samples: int = 400,
) -> tuple[float, float]:
    """Least-squares fit of the mandated form's gamma against the canonical curve.

    Returns ``(gamma, rmse_m_s)``. Pure Python grid search plus golden-section
    refinement -- no SciPy, and deterministic, so the shipped constant is
    reproducible in CI.
    """
    lo, hi = band
    grid = [lo + (hi - lo) * i / (samples - 1) for i in range(samples)]
    target = [weidmann_canonical_speed(r) for r in grid]

    def sse(gamma: float) -> float:
        return sum(
            (weidmann_speed(r, gamma=gamma) - t) ** 2 for r, t in zip(grid, target)
        )

    if candidates is None:
        a, b = 1e-3, 5.0
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b - phi * (b - a), a + phi * (b - a)
        for _ in range(200):
            if sse(c) < sse(d):
                b, d = d, c
                c = b - phi * (b - a)
            else:
                a, c = c, d
                d = a + phi * (b - a)
            if abs(b - a) < 1e-9:
                break
        best = (a + b) / 2.0
    else:
        best = min(candidates, key=sse)

    rmse = math.sqrt(sse(best) / len(grid))
    return best, rmse

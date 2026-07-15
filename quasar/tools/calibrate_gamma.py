"""Fit the mandated speed-density form's gamma to the canonical Weidmann curve.

Run from the repo root:

    PYTHONPATH=src python3 tools/calibrate_gamma.py

Prints the fitted constant, the fit error, and a residual table so the
divergence between the two forms is visible rather than hidden.
"""

from __future__ import annotations

from quasar.crowd import (
    GAMMA_MANDATED,
    OPERATIONAL_BAND,
    fit_gamma,
    level_of_service,
    weidmann_canonical_speed,
    weidmann_speed,
)


def main() -> None:
    gamma, rmse = fit_gamma()
    print(f"fitted gamma      : {gamma:.4f} m^2/ped")
    print(f"shipped gamma     : {GAMMA_MANDATED:.4f} m^2/ped")
    print(f"fit RMSE          : {rmse:.4f} m/s over rho in {OPERATIONAL_BAND}")
    print()
    print(f"{'rho':>6} {'LOS':>4} {'mandated':>9} {'canonical':>10} {'residual':>9}")
    for i in range(0, 28):
        rho = 0.2 + i * 0.2
        if rho > OPERATIONAL_BAND[1]:
            break
        m = weidmann_speed(rho, gamma=gamma)
        c = weidmann_canonical_speed(rho)
        print(
            f"{rho:6.2f} {level_of_service(rho).value:>4} "
            f"{m:9.3f} {c:10.3f} {m - c:+9.3f}"
        )


if __name__ == "__main__":
    main()

"""Nodal equibiaxial laminate equilibrium and delamination energy from the original course driver."""

from __future__ import annotations
import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _solve_abd_nodal(
    x_arr: np.ndarray, M_arr: np.ndarray, eps_star: np.ndarray
) -> tuple[float, float]:
    """
    Solve the free-laminate ABD system for a nodal mismatch field.

        [A  B] [ε0]   [N_mis]
        [B  D] [κ ]   [M_mis]

    where (all integrals via trapezoidal rule):
        A = ∫ M dx,   B = ∫ M x dx,   D = ∫ M x² dx
        N_mis = ∫ M ε* dx,   M_mis = ∫ M ε* x dx

    Source: HW1/MATRL263HW1.py :: _solve_free_laminate()
    Returns (ε0, κ).
    """
    A = float(_trapz(M_arr, x_arr))
    B = float(_trapz(M_arr * x_arr, x_arr))
    D = float(_trapz(M_arr * x_arr**2, x_arr))
    Nmis = float(_trapz(M_arr * eps_star, x_arr))
    Mmis = float(_trapz(M_arr * eps_star * x_arr, x_arr))
    K = np.array([[A, B], [B, D]], dtype=float)
    if abs(np.linalg.det(K)) < 1e-18:
        raise ValueError("ABD matrix nearly singular.")
    eps0, kappa = np.linalg.solve(K, [Nmis, Mmis])
    return float(eps0), float(kappa)


def _elastic_energy_nodal(
    x_arr: np.ndarray,
    M_arr: np.ndarray,
    eps_star: np.ndarray,
    eps0: float,
    kappa: float,
) -> float:
    """
    Elastic energy per unit area:
        U = ∫ M(x) · [ε0 + κ x - ε*(x)]² dx

    For equibiaxial plane stress the energy density is W = M ε_el² (not ½ M ε_el²)
    because both in-plane stress components contribute equally:
        W = ½(σ_x ε_x^el + σ_y ε_y^el) = σ ε_el = M ε_el²
    Source: comment in HW1/MATRL263HW1.py :: _elastic_energy_per_unit_area()
    """
    eps_el = eps0 + kappa * x_arr - eps_star
    return float(_trapz(M_arr * eps_el**2, x_arr))


def compute_stress_and_ERR(
    x_arr: np.ndarray, M_arr: np.ndarray, eps_star: np.ndarray, x_int: float
) -> dict:
    """
    For a given nodal mismatch field eps_star on grid x_arr:
      1. Solve the free-laminate ABD system  →  (ε0, κ).
      2. Through-thickness biaxial stress:  σ(x) = M(x) · [ε0 + κ x - ε*(x)].
      3. ERR for delamination at x_int:
             G = U_bonded - (U_substrate_free + U_film_free)
         Source: HW1/MATRL263HW1.py :: multilayer_stress_curvature_delamination() docstring.

    Returns dict: eps0, kappa, sigma, U_bonded, U_sub, U_film, G
    """
    # Full bonded bilayer
    eps0, kappa = _solve_abd_nodal(x_arr, M_arr, eps_star)
    sigma = M_arr * (eps0 + kappa * x_arr - eps_star)
    U_bonded = _elastic_energy_nodal(x_arr, M_arr, eps_star, eps0, kappa)

    # Index of node nearest to the interface
    i_int = int(np.argmin(np.abs(x_arr - x_int)))

    # Substrate sub-laminate (nodes 0 .. i_int, inclusive)
    xs = x_arr[: i_int + 1]
    Ms = M_arr[: i_int + 1]
    es = eps_star[: i_int + 1]
    e0s, ks = _solve_abd_nodal(xs, Ms, es)
    U_sub = _elastic_energy_nodal(xs, Ms, es, e0s, ks)

    # Film sub-laminate (nodes i_int .. end)
    xf = x_arr[i_int:]
    Mf = M_arr[i_int:]
    ef = eps_star[i_int:]
    e0f, kf = _solve_abd_nodal(xf, Mf, ef)
    U_film = _elastic_energy_nodal(xf, Mf, ef, e0f, kf)

    G = U_bonded - (U_sub + U_film)

    return dict(
        eps0=eps0,
        kappa=kappa,
        sigma=sigma,
        U_bonded=U_bonded,
        U_sub=U_sub,
        U_film=U_film,
        G=G,
    )

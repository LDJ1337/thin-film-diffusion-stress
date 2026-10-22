import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class LayerStressResult:
    layer: int
    z_bottom: float
    z_top: float
    z_mid: float
    sigma_bottom: float
    sigma_mid: float
    sigma_top: float
    sigma_avg: float
    biaxial_modulus: float
    mismatch_strain: float


def _validate_inputs(
    N: int,
    thicknesses: np.ndarray,
    moduli: np.ndarray,
    poissons: np.ndarray,
    mismatch_strains: np.ndarray,
) -> None:
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    arrays = [thicknesses, moduli, poissons, mismatch_strains]
    names = ["thicknesses", "moduli", "poissons", "mismatch_strains"]

    for arr, name in zip(arrays, names):
        if len(arr) != N:
            raise ValueError(f"{name} must have length N = {N}.")

    if np.any(thicknesses <= 0):
        raise ValueError("All thicknesses must be positive.")

    if np.any(moduli <= 0):
        raise ValueError("All elastic moduli must be positive.")

    if np.any((poissons <= -1.0) | (poissons >= 0.5)):
        raise ValueError(
            "Poisson's ratios must satisfy -1 < nu < 0.5 for isotropic elasticity."
        )


def _layer_boundaries(thicknesses: np.ndarray) -> np.ndarray:
    """
    Returns z-coordinates of all layer boundaries with z=0 at the bottom surface.
    """
    z = np.zeros(len(thicknesses) + 1)
    z[1:] = np.cumsum(thicknesses)
    return z


def _biaxial_modulus(E: np.ndarray, nu: np.ndarray) -> np.ndarray:
    return E / (1.0 - nu)


def _solve_free_laminate(
    thicknesses: np.ndarray,
    moduli: np.ndarray,
    poissons: np.ndarray,
    mismatch_strains: np.ndarray,
) -> Dict[str, Any]:
    """
    Solve for membrane strain epsilon0 and curvature kappa of a free multilayer laminate.

    Coordinate system:
        z = 0 at bottom surface, z increases upward through thickness.

    Governing equations for a free laminate:
        A*eps0 + B*kappa = Nmis
        B*eps0 + D*kappa = Mmis
    """
    N = len(thicknesses)
    z = _layer_boundaries(thicknesses)
    M = _biaxial_modulus(moduli, poissons)

    A = 0.0
    B = 0.0
    D = 0.0
    Nmis = 0.0
    Mmis = 0.0

    for i in range(N):
        z0 = z[i]
        z1 = z[i + 1]
        h = z1 - z0
        dz2 = z1**2 - z0**2
        dz3 = z1**3 - z0**3

        A += M[i] * h
        B += M[i] * 0.5 * dz2
        D += M[i] * (1.0 / 3.0) * dz3

        Nmis += M[i] * mismatch_strains[i] * h
        Mmis += M[i] * mismatch_strains[i] * 0.5 * dz2

    K = np.array([[A, B], [B, D]], dtype=float)
    rhs = np.array([Nmis, Mmis], dtype=float)

    if abs(np.linalg.det(K)) < 1e-18:
        raise ValueError("Laminate stiffness matrix is singular or nearly singular.")

    epsilon0, kappa = np.linalg.solve(K, rhs)

    return {
        "epsilon0": float(epsilon0),
        "kappa": float(kappa),
        "z_boundaries": z,
        "biaxial_moduli": M,
        "A": float(A),
        "B": float(B),
        "D": float(D),
        "Nmis": float(Nmis),
        "Mmis": float(Mmis),
    }


def _stress_in_layer(
    Mi: float, eps0: float, kappa: float, eps_m: float, z: float
) -> float:
    return Mi * (eps0 + kappa * z - eps_m)


def _elastic_energy_per_unit_area(
    thicknesses: np.ndarray,
    moduli: np.ndarray,
    poissons: np.ndarray,
    mismatch_strains: np.ndarray,
) -> float:
    """
    Strain energy per unit area for the relaxed free laminate state.
    """
    sol = _solve_free_laminate(thicknesses, moduli, poissons, mismatch_strains)
    eps0 = sol["epsilon0"]
    kappa = sol["kappa"]
    z = sol["z_boundaries"]
    M = sol["biaxial_moduli"]

    U = 0.0
    for i in range(len(thicknesses)):
        z0 = z[i]
        z1 = z[i + 1]
        e = mismatch_strains[i]
        a = eps0 - e

        h = z1 - z0
        dz2 = z1**2 - z0**2
        dz3 = z1**3 - z0**3

        # For equal biaxial plane stress, the strain energy density is
        # W = σ²/M = M·ε_el²  (NOT ½·M·ε_el² as in uniaxial).
        # Both in-plane stress components contribute:
        #   W = ½(σ_x ε_x^el + σ_y ε_y^el) = ½·2σ·(σ/M) = σ²/M
        # U_i = ∫ M_i (a + kappa z)^2 dz
        Ui = M[i] * (a**2 * h + a * kappa * dz2 + (kappa**2 / 3.0) * dz3)
        U += Ui

    return float(U)


def multilayer_stress_curvature_delamination(
    N: int,
    thicknesses: List[float],
    moduli: List[float],
    poissons: List[float],
    mismatch_strains: List[float],
) -> Dict[str, Any]:
    """
    Main user-facing function.

    Parameters
    ----------
    N : int
        Number of layers.
    thicknesses : list of float
        Layer thicknesses [m].
    moduli : list of float
        Young's moduli [Pa].
    poissons : list of float
        Poisson's ratios [-].
    mismatch_strains : list of float
        Free mismatch strains in each layer [-].
        Examples:
            thermal: alpha_i * DeltaT
            chemical swelling/shrinkage
            intrinsic process strain

    Returns
    -------
    dict
        {
            "curvature": ...,
            "membrane_strain": ...,
            "total_thickness": ...,
            "layer_stresses": [...],
            "energy_release_rates": [...]
        }

    Notes
    -----
    The delamination driving force at interface j is computed as

        G_j = U_bonded - (U_top_free + U_bottom_free)

    where U_bonded is the elastic energy per unit area of the intact laminate,
    and U_top_free / U_bottom_free are the elastic energies per unit area of the
    two sublaminates after complete interfacial separation and full relaxation.

    Positive G_j means separation lowers elastic energy and is energetically favorable.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    moduli = np.asarray(moduli, dtype=float)
    poissons = np.asarray(poissons, dtype=float)
    mismatch_strains = np.asarray(mismatch_strains, dtype=float)

    _validate_inputs(N, thicknesses, moduli, poissons, mismatch_strains)

    sol = _solve_free_laminate(thicknesses, moduli, poissons, mismatch_strains)
    eps0 = sol["epsilon0"]
    kappa = sol["kappa"]
    z = sol["z_boundaries"]
    M = sol["biaxial_moduli"]

    layer_results = []
    for i in range(N):
        z0 = z[i]
        z1 = z[i + 1]
        zm = 0.5 * (z0 + z1)

        sigma_bottom = _stress_in_layer(M[i], eps0, kappa, mismatch_strains[i], z0)
        sigma_mid = _stress_in_layer(M[i], eps0, kappa, mismatch_strains[i], zm)
        sigma_top = _stress_in_layer(M[i], eps0, kappa, mismatch_strains[i], z1)
        sigma_avg = _stress_in_layer(M[i], eps0, kappa, mismatch_strains[i], zm)

        layer_results.append(
            LayerStressResult(
                layer=i + 1,
                z_bottom=float(z0),
                z_top=float(z1),
                z_mid=float(zm),
                sigma_bottom=float(sigma_bottom),
                sigma_mid=float(sigma_mid),
                sigma_top=float(sigma_top),
                sigma_avg=float(sigma_avg),
                biaxial_modulus=float(M[i]),
                mismatch_strain=float(mismatch_strains[i]),
            )
        )

    U_bonded = _elastic_energy_per_unit_area(
        thicknesses, moduli, poissons, mismatch_strains
    )

    G_interfaces = []
    for j in range(1, N):
        # Interface j lies between layers j and j+1 in 1-based indexing.
        t_bot = thicknesses[:j]
        E_bot = moduli[:j]
        nu_bot = poissons[:j]
        em_bot = mismatch_strains[:j]

        t_top = thicknesses[j:]
        E_top = moduli[j:]
        nu_top = poissons[j:]
        em_top = mismatch_strains[j:]

        U_bot = _elastic_energy_per_unit_area(t_bot, E_bot, nu_bot, em_bot)
        U_top = _elastic_energy_per_unit_area(t_top, E_top, nu_top, em_top)

        G = U_bonded - (U_bot + U_top)

        G_interfaces.append(
            {
                "interface": j,  # between layer j and layer j+1
                "between_layers": (j, j + 1),
                "energy_release_rate": float(G),  # J/m^2
            }
        )

    return {
        "curvature": float(kappa),  # 1/m
        "membrane_strain": float(eps0),  # strain at z=0 in this coordinate choice
        "total_thickness": float(np.sum(thicknesses)),
        "bonded_energy_per_unit_area": float(U_bonded),  # J/m^2
        "layer_stresses": [asdict(r) for r in layer_results],
        "energy_release_rates": G_interfaces,
    }


if __name__ == "__main__":
    # Example: 3-layer stack
    N = 3
    thicknesses = [200e-6, 1.0e-6, 2.0e-6]  # m
    moduli = [150e9, 250e9, 180e9]  # Pa
    poissons = [0.22, 0.27, 0.25]  # -
    mismatch_strains = [0.0, -2.0e-3, 1.0e-3]  # -

    result = multilayer_stress_curvature_delamination(
        N, thicknesses, moduli, poissons, mismatch_strains
    )

    print("\n=== GLOBAL RESPONSE ===")
    print(f"Curvature kappa = {result['curvature']:.6e} 1/m")
    print(f"Bonded elastic energy = {result['bonded_energy_per_unit_area']:.6e} J/m^2")

    print("\n=== LAYER STRESSES ===")
    for r in result["layer_stresses"]:
        print(
            f"Layer {r['layer']}: "
            f"sigma_bottom={r['sigma_bottom']:.6e} Pa, "
            f"sigma_mid={r['sigma_mid']:.6e} Pa, "
            f"sigma_top={r['sigma_top']:.6e} Pa"
        )

    print("\n=== INTERFACIAL DELAMINATION DRIVING FORCES ===")
    for g in result["energy_release_rates"]:
        print(
            f"Interface {g['interface']} "
            f"(between layers {g['between_layers'][0]} and {g['between_layers'][1]}): "
            f"G = {g['energy_release_rate']:.6e} J/m^2"
        )

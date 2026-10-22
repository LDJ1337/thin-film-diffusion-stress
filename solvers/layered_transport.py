"""
MATRL263HW2 — transient diffusion / conduction through N planar layers.

∂c/∂t = ∂(D ∂c/∂x)/∂x + S(x)

Piecewise-constant D(x), optional volumetric source S (per-node, constant in time).

Interior interfaces: continuity of c and of D ∂c/∂x (handled with harmonic face
diffusivity on a uniform grid).

Boundary conditions:
  ("dirichlet", value) — fixed concentration
  ("neumann", dcdx)  — fixed ∂c/∂x (e.g. 0 for insulated / symmetry)

Heat conduction (thermal analogue): see solve_layered_heat_conduction for
ρ c_p ∂T/∂t = ∂/∂x ( k ∂T/∂x ) with harmonic face conductivity so k ∂T/∂x is
continuous across perfect interfaces (Begley & Hutchinson §2.6 / §15.2 framework).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Sequence, Tuple, Union

import numpy as np

BCSpec = Tuple[Literal["dirichlet", "neumann"], float]

HeatLeftBC = Tuple[Literal["dirichlet"], float]
HeatRightBC = Union[
    Tuple[Literal["dirichlet"], float],
    Tuple[Literal["neumann"], float],
    Tuple[Literal["robin"], float, Callable[[float], float]],
]


@dataclass
class LayeredDiffusionResult:
    x: np.ndarray
    times: np.ndarray
    concentrations: np.ndarray  # shape (n_times, n_points)


def piecewise_constant_on_layers(
    x: np.ndarray,
    layer_thicknesses: Sequence[float],
    layer_values: Sequence[float],
) -> np.ndarray:
    """Assign one scalar property per layer to each node (same indexing as diffusivity_on_grid)."""
    h = np.asarray(layer_thicknesses, dtype=float)
    vals = np.asarray(layer_values, dtype=float)
    if np.any(h <= 0):
        raise ValueError("Layer thicknesses must be positive.")
    edges = np.concatenate([[0.0], np.cumsum(h)])
    idx = np.searchsorted(edges[1:], x, side="right")
    idx = np.clip(idx, 0, len(vals) - 1)
    return vals[idx]


def diffusivity_on_grid(
    x: np.ndarray,
    layer_thicknesses: Sequence[float],
    layer_diffusivities: Sequence[float],
) -> np.ndarray:
    """Assign piecewise-constant D to each node (x increases through the stack)."""
    h = np.asarray(layer_thicknesses, dtype=float)
    Dl = np.asarray(layer_diffusivities, dtype=float)
    if np.any(h <= 0):
        raise ValueError("Layer thicknesses must be positive.")
    if np.any(Dl <= 0):
        raise ValueError("Layer diffusivities must be positive.")
    return piecewise_constant_on_layers(x, h, Dl)


def _face_diffusivity(D: np.ndarray) -> np.ndarray:
    """Harmonic means D_{i-1/2} between adjacent nodes; length n+1, index i = face left of node i."""
    n = D.size - 1
    out = np.zeros(n + 1)
    out[0] = D[0]  # unused for standard stencil
    for k in range(1, n + 1):
        a, b = D[k - 1], D[k]
        out[k] = 2.0 * a * b / (a + b) if (a + b) > 0 else 0.0
    return out


def _thomas_solve(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for tridiagonal system with arrays of length n_unknowns."""
    n = rhs.size
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / denom if i < n - 1 else 0.0
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_layered_diffusion(
    *,
    layer_thicknesses: Sequence[float],
    layer_diffusivities: Sequence[float],
    nx: int,
    t_end: float,
    dt: float,
    c_initial: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    left_bc: BCSpec,
    right_bc: BCSpec,
    source: Union[None, np.ndarray, Callable[[np.ndarray, float], np.ndarray]] = None,
    save_times: Union[None, Sequence[float]] = None,
) -> LayeredDiffusionResult:
    """
    Backward Euler on a uniform mesh (unconditionally stable). The scalar `dt`
    is the **actual** time step (the final step is shortened to hit `t_end`).
    Accuracy in fast transients improves with smaller `dt`; order O(dt).

    `save_times`: optional sorted snapshot times; `0` is prepended if missing.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")
    if not np.isfinite(t_end) or t_end < 0:
        raise ValueError("t_end must be nonnegative and finite.")
    h = np.asarray(layer_thicknesses, dtype=float)
    L = float(np.sum(h))
    if nx < 3:
        raise ValueError("nx must be at least 3.")
    if t_end <= 0:
        x0 = np.linspace(0.0, L, nx)
        if callable(c_initial):
            c0 = c_initial(x0).astype(float)
        else:
            c0 = np.asarray(c_initial, dtype=float).copy()
            if c0.shape != x0.shape:
                raise ValueError("c_initial must match nx.")
        return LayeredDiffusionResult(
            x=x0,
            times=np.array([0.0]),
            concentrations=c0.reshape(1, -1),
        )

    x = np.linspace(0.0, L, nx)
    dx = float(x[1] - x[0])
    D = diffusivity_on_grid(x, h, layer_diffusivities)
    D_face = _face_diffusivity(D)

    if callable(c_initial):
        c = c_initial(x).astype(float).copy()
    else:
        c = np.asarray(c_initial, dtype=float).copy()
        if c.shape != x.shape:
            raise ValueError("c_initial must match nx.")

    if save_times is None:
        nframes = min(10, max(2, int(t_end / dt) + 1))
        save_times = np.linspace(0.0, t_end, nframes)
    save_times = np.unique(np.asarray(save_times, dtype=float))
    if (
        np.any(~np.isfinite(save_times))
        or np.any(save_times < 0)
        or np.any(save_times > t_end)
    ):
        raise ValueError("save_times must be finite and lie between zero and t_end.")
    save_times.sort()
    if save_times.size == 0 or save_times[0] > 1e-18:
        save_times = np.concatenate([[0.0], save_times])

    # source at nodes
    def _S(xx: np.ndarray, t: float) -> np.ndarray:
        if source is None:
            return np.zeros_like(xx)
        if callable(source):
            return np.asarray(source(xx, t), dtype=float)
        return np.asarray(source, dtype=float)

    out_t: List[float] = []
    out_c: List[np.ndarray] = []

    t = 0.0
    out_t.append(0.0)
    out_c.append(c.copy())

    target_idx = 1
    while t < t_end - 1e-18:
        dt_use = min(float(dt), t_end - t)
        # Land on every requested snapshot rather than labeling a later state
        # with an earlier requested time.
        if target_idx < save_times.size:
            dt_use = min(dt_use, float(save_times[target_idx]) - t)
        t += dt_use

        S_vec = _S(x, t)

        # unknown indexing
        n = nx
        lo = np.zeros(n)
        di = np.ones(n)
        up = np.zeros(n)
        rh = np.zeros(n)

        # interior Laplacian coefficients
        lam = D_face * dt_use / (dx * dx)
        for i in range(1, n - 1):
            di[i] = 1.0 + lam[i] + lam[i + 1]
            lo[i] = -lam[i]
            up[i] = -lam[i + 1]
            rh[i] = c[i] + dt_use * S_vec[i]

        # left boundary i = 0
        lt, lv = left_bc
        if lt == "dirichlet":
            di[0] = 1.0
            lo[0] = 0.0
            up[0] = 0.0
            rh[0] = lv
            if n > 1:
                lo[1] = 0.0
                rh[1] += lam[1] * lv
        else:  # neumann: dcdx = lv at x=0  => one-sided / ghost
            lam0 = D_face[1] * dt_use / (dx * dx)
            di[0] = 1.0 + 2.0 * lam0
            lo[0] = 0.0
            up[0] = -2.0 * lam0
            rh[0] = c[0] + 2.0 * lam0 * (lv * dx) + dt_use * S_vec[0]

        # right boundary i = n-1
        rt, rv = right_bc
        if rt == "dirichlet":
            di[-1] = 1.0
            lo[-1] = 0.0
            up[-1] = 0.0
            rh[-1] = rv
            if n > 1:
                up[-2] = 0.0
                rh[-2] += lam[-1] * rv
        else:
            lamN = D_face[-1] * dt_use / (dx * dx)
            di[-1] = 1.0 + 2.0 * lamN
            lo[-1] = -2.0 * lamN
            up[-1] = 0.0
            rh[-1] = c[-1] + 2.0 * lamN * (rv * dx) + dt_use * S_vec[-1]

        # mask fixed rows out of solve: eliminate dirichlet rows from tridiagonal subset
        free = np.ones(n, dtype=bool)
        if lt == "dirichlet":
            free[0] = False
        if rt == "dirichlet":
            free[-1] = False
        idx = np.nonzero(free)[0]
        m = idx.size
        if m == 0:
            c_new = rh.copy()
        else:
            Lm = np.zeros(m)
            Dm = np.zeros(m)
            Um = np.zeros(m)
            Rm = np.zeros(m)
            for k, i in enumerate(idx):
                Dm[k] = di[i]
                Rm[k] = rh[i]
                if i > 0 and free[i - 1]:
                    Lm[k] = lo[i]
                elif i > 0:
                    # neighbor Dirichlet already folded into rh
                    pass
                if i < n - 1 and free[i + 1]:
                    Um[k] = up[i]
            sol = _thomas_solve(Lm, Dm, Um, Rm)
            c_new = rh.copy()
            for k, i in enumerate(idx):
                c_new[i] = sol[k]
            if lt == "dirichlet":
                c_new[0] = lv
            if rt == "dirichlet":
                c_new[-1] = rv

        c = c_new

        while target_idx < save_times.size and t + 1e-15 >= save_times[target_idx]:
            out_t.append(float(save_times[target_idx]))
            out_c.append(c.copy())
            target_idx += 1

    times = np.asarray(out_t)
    conc = np.vstack(out_c)
    return LayeredDiffusionResult(x=x, times=times, concentrations=conc)


def solve_layered_heat_conduction(
    *,
    layer_thicknesses: Sequence[float],
    layer_k: Sequence[float],
    layer_rho_cp: Sequence[float],
    nx: int,
    t_end: float,
    dt: float,
    T_initial: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    left_bc: HeatLeftBC,
    right_bc: HeatRightBC,
    save_times: Union[None, Sequence[float]] = None,
) -> LayeredDiffusionResult:
    """
    Transient conduction ρ c_p ∂T/∂t = ∂/∂x ( k ∂T/∂x ) with piecewise-constant k and
    ρ c_p (volumetric heat capacity, J/(m^3·K)).

    Face conductivities are harmonic means of nodal k (continuous heat flux).

    Boundary conditions:
      left: ("dirichlet", Tb)
      right: ("dirichlet", T) or ("robin", h, T_inf(t)) with k ∂T/∂x|_surf = h (T − T_inf(t)),
             matching Begley & Hutchinson’s GetThermalTQ convention (their k T' = h (T − T_∞)).

    Returns nodal temperatures in ``concentrations`` (same dataclass field as diffusion).
    """
    hlay = np.asarray(layer_thicknesses, dtype=float)
    L = float(np.sum(hlay))
    if nx < 3:
        raise ValueError("nx must be at least 3.")
    k_l = np.asarray(layer_k, dtype=float)
    rcp_l = np.asarray(layer_rho_cp, dtype=float)
    if np.any(k_l <= 0) or np.any(rcp_l <= 0):
        raise ValueError(
            "Thermal conductivity and ρ c_p must be positive in every layer."
        )

    if t_end <= 0:
        x0 = np.linspace(0.0, L, nx)
        if callable(T_initial):
            T0 = T_initial(x0).astype(float)
        else:
            T0 = np.asarray(T_initial, dtype=float).copy()
            if T0.shape != x0.shape:
                raise ValueError("T_initial must match nx.")
        return LayeredDiffusionResult(
            x=x0,
            times=np.array([0.0]),
            concentrations=T0.reshape(1, -1),
        )

    x = np.linspace(0.0, L, nx)
    dx = float(x[1] - x[0])
    k = piecewise_constant_on_layers(x, hlay, k_l)
    rho_cp = piecewise_constant_on_layers(x, hlay, rcp_l)
    k_face = _face_diffusivity(k)

    if callable(T_initial):
        T = T_initial(x).astype(float).copy()
    else:
        T = np.asarray(T_initial, dtype=float).copy()
        if T.shape != x.shape:
            raise ValueError("T_initial must match nx.")

    if save_times is None:
        nframes = min(16, max(2, int(t_end / dt) + 1))
        save_times = np.linspace(0.0, t_end, nframes)
    save_times = np.unique(np.asarray(save_times, dtype=float))
    save_times.sort()
    if save_times.size == 0 or save_times[0] > 1e-18:
        save_times = np.concatenate([[0.0], save_times])

    lt, Tb = left_bc
    if lt != "dirichlet":
        raise ValueError(
            "Heat solver currently requires a Dirichlet bottom (left) surface."
        )

    out_t: List[float] = []
    out_T: List[np.ndarray] = []

    t = 0.0
    out_t.append(0.0)
    out_T.append(T.copy())

    target_idx = 1
    n = nx
    while t < t_end - 1e-18:
        dt_use = min(float(dt), t_end - t)
        t += dt_use

        rt = right_bc[0]
        Tr_val = float(right_bc[1]) if rt == "dirichlet" else None

        lo = np.zeros(n)
        di = np.zeros(n)
        up = np.zeros(n)
        rh = np.zeros(n)

        # Unknown rows 1 .. n−2 always (coupled stencil); Robin adds unknown n−1
        for i in range(1, n - 1):
            kim = k_face[i]
            kip = k_face[i + 1]
            rho_i = rho_cp[i]
            lo_i = -kim / (dx * dx)
            up_i = -kip / (dx * dx)
            diag_i = rho_i / dt_use + (kim + kip) / (dx * dx)
            rh_i = rho_i * T[i] / dt_use

            if i == 1:
                lo_i = 0.0
                rh_i += kim * Tb / (dx * dx)

            if rt == "dirichlet" and i == n - 2:
                up_i = 0.0
                rh_i += kip * float(Tr_val) / (dx * dx)

            lo[i] = lo_i
            up[i] = up_i
            di[i] = diag_i
            rh[i] = rh_i

        di[0] = 1.0
        lo[0] = 0.0
        up[0] = 0.0
        rh[0] = Tb

        if rt == "dirichlet":
            di[-1] = 1.0
            lo[-1] = 0.0
            up[-1] = 0.0
            rh[-1] = float(Tr_val)
        elif rt == "robin":
            hconv, T_inf_fn = right_bc[1], right_bc[2]
            T_inf = float(T_inf_fn(t))
            kim = k_face[n - 1]
            di[-1] = rho_cp[n - 1] / dt_use + kim / (dx * dx) + hconv / dx
            lo[-1] = -kim / (dx * dx)
            up[-1] = 0.0
            rh[-1] = rho_cp[n - 1] * T[n - 1] / dt_use + hconv * T_inf / dx
        else:
            raise ValueError("Right BC must be dirichlet or robin for heat conduction.")

        free = np.ones(n, dtype=bool)
        free[0] = False
        if rt == "dirichlet":
            free[-1] = False

        idx = np.nonzero(free)[0]
        m = idx.size
        if m == 0:
            T_new = rh.copy()
        else:
            Lm = np.zeros(m)
            Dm = np.zeros(m)
            Um = np.zeros(m)
            Rm = np.zeros(m)
            for krow, i in enumerate(idx):
                Dm[krow] = di[i]
                Rm[krow] = rh[i]
                if i > 0 and free[i - 1]:
                    Lm[krow] = lo[i]
                if i < n - 1 and free[i + 1]:
                    Um[krow] = up[i]
            sol = _thomas_solve(Lm, Dm, Um, Rm)
            T_new = rh.copy()
            for krow, i in enumerate(idx):
                T_new[i] = sol[krow]
            T_new[0] = Tb
            if rt == "dirichlet":
                T_new[-1] = float(Tr_val)

        T = T_new

        while target_idx < save_times.size and t + 1e-15 >= save_times[target_idx]:
            out_t.append(float(save_times[target_idx]))
            out_T.append(T.copy())
            target_idx += 1

    times = np.asarray(out_t)
    temp = np.vstack(out_T)
    return LayeredDiffusionResult(x=x, times=times, concentrations=temp)


def flux_at_nodes(x: np.ndarray, D: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Fick flux J = -D ∂c/∂x (central differences), shape same as x."""
    dx = x[1] - x[0]
    J = np.zeros_like(c)
    J[1:-1] = -D[1:-1] * (c[2:] - c[:-2]) / (2.0 * dx)
    J[0] = -D[0] * (c[1] - c[0]) / dx
    J[-1] = -D[-1] * (c[-1] - c[-2]) / dx
    return J


def interface_flux_jump(
    x: np.ndarray,
    layer_thicknesses: Sequence[float],
    D_layers: Sequence[float],
    c: np.ndarray,
    interface_index: int,
) -> float:
    """
    Relative flux mismatch at interface `interface_index` (1 .. N_layers-1)
    using one-sided derivatives. Should be ≪ 1 when scaled by |J|.
    """
    h = np.asarray(layer_thicknesses, dtype=float)
    z_int = float(np.sum(h[:interface_index]))
    k = int(np.searchsorted(x, z_int, side="left"))
    k = max(1, min(k, x.size - 2))

    D = diffusivity_on_grid(x, h, D_layers)
    dx = x[1] - x[0]
    Jp = -D[k] * (c[k + 1] - c[k]) / dx
    Jm = -D[k - 1] * (c[k] - c[k - 1]) / dx
    ref = max(abs(Jp), abs(Jm), 1e-30)
    return abs(Jp - Jm) / ref


def heat_flux_jump_at_interface(
    x: np.ndarray,
    layer_thicknesses: Sequence[float],
    layer_k: Sequence[float],
    T: np.ndarray,
    interface_index: int,
) -> float:
    """
    Relative mismatch of conductive heat flux q = −k ∂T/∂x across interface
    ``interface_index`` (1 .. N_layers−1), using one-sided derivatives.
    """
    hlay = np.asarray(layer_thicknesses, dtype=float)
    z_int = float(np.sum(hlay[:interface_index]))
    k = int(np.searchsorted(x, z_int, side="left"))
    k = max(1, min(k, x.size - 2))

    kk = piecewise_constant_on_layers(x, hlay, layer_k)
    dx = x[1] - x[0]
    qp = -kk[k] * (T[k + 1] - T[k]) / dx
    qm = -kk[k - 1] * (T[k] - T[k - 1]) / dx
    ref = max(abs(qp), abs(qm), 1e-30)
    return abs(qp - qm) / ref


def bh_ambient_temperature_cycle_degC(
    t: float, a: float, T_room: float = 20.0
) -> float:
    """
    Ambient temperature history used in Begley & Hutchinson §15.5 / Fig. 15.2:

        T_∞(t) = T_room + 500 * φ(t),

    where φ ramps linearly up (slope ``a``) for 1/a seconds, holds at 1 for 2 seconds,
    then ramps down to 0 over 1/a seconds (same ``a``). The same form appears in their
    Mathematica snippet for ``T[a]``.
    """
    if t <= 0.0:
        return float(T_room)
    t_up = 1.0 / a
    t_hold_end = 2.0 + 1.0 / a
    t_down_end = 2.0 + 2.0 / a
    if t <= t_up:
        phi = a * t
    elif t <= t_hold_end:
        phi = 1.0
    elif t <= t_down_end:
        phi = 1.0 - a * (t - t_hold_end)
    else:
        phi = 0.0
    return float(T_room + 500.0 * phi)

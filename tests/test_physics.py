import numpy as np
import pytest
from solvers.layered_transport import solve_layered_diffusion as solve
from solvers.layered_transport import piecewise_constant_on_layers
from solvers.coupled_stress import compute_stress_and_ERR
from solvers.laminate_stress import multilayer_stress_curvature_delamination
from run_model import simulate, save_results


def transport(**overrides):
    params = dict(
        layer_thicknesses=[1.0],
        layer_diffusivities=[1.0],
        nx=101,
        t_end=0.1,
        dt=0.001,
        c_initial=lambda x: np.ones_like(x),
        left_bc=("neumann", 0.0),
        right_bc=("neumann", 0.0),
        save_times=[0.0, 0.1],
    )
    params.update(overrides)
    return solve(**params)


def test_uniform_field_is_stationary():
    result = transport()
    np.testing.assert_allclose(result.concentrations, 1.0, atol=1e-12)


def test_closed_system_conserves_inventory():
    r = transport(c_initial=lambda x: 1 + 0.2 * np.cos(2 * np.pi * x))
    integral = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    mass = integral(r.concentrations, r.x, axis=1)
    np.testing.assert_allclose(mass, mass[0], rtol=1e-10)


def test_homogeneous_mode_matches_analytical_solution():
    r = transport(
        c_initial=lambda x: np.cos(np.pi * x / 2),
        right_bc=("dirichlet", 0.0),
        dt=0.0005,
        t_end=0.05,
        save_times=[0.0, 0.0125, 0.05],
    )
    expected = np.cos(np.pi * r.x / 2) * np.exp(-(np.pi**2) * 0.05 / 4)
    np.testing.assert_allclose(r.concentrations[-1], expected, atol=3e-4)


def test_snapshots_land_on_requested_times():
    # A uniform source gives c=t exactly, exposing any relabeled later state.
    r = transport(
        nx=5,
        dt=2.0,
        t_end=2.0,
        c_initial=lambda x: np.zeros_like(x),
        source=lambda x, t: np.ones_like(x),
        save_times=[0.0, 0.5, 1.25, 2.0],
    )
    np.testing.assert_allclose(r.times, [0.0, 0.5, 1.25, 2.0])
    np.testing.assert_allclose(
        r.concentrations,
        np.broadcast_to(r.times[:, None], r.concentrations.shape),
        atol=1e-12,
    )


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan])
def test_invalid_dt_rejected(dt):
    with pytest.raises(ValueError, match="dt"):
        transport(dt=dt)


@pytest.mark.parametrize("strain", [0.0, 2e-3])
def test_uniform_eigenstrain_has_no_stress_or_delamination_energy(strain):
    x = np.linspace(0, 22e-6, 441)
    M = piecewise_constant_on_layers(x, [20e-6, 2e-6], [200e9 / 0.7, 100e9 / 0.7])
    result = compute_stress_and_ERR(x, M, np.full_like(x, strain), 20e-6)
    assert np.max(np.abs(result["sigma"])) < 1e-3
    assert abs(result["G"]) < 1e-12


def test_piecewise_strain_converges_toward_analytic_layer_solution():
    exact = multilayer_stress_curvature_delamination(
        2, [20e-6, 2e-6], [200e9, 100e9], [0.3, 0.3], [0.0, -1e-3]
    )
    reference = exact["energy_release_rates"][0]["energy_release_rate"]
    errors = []
    for n in [221, 441, 881]:
        x = np.linspace(0, 22e-6, n)
        M = piecewise_constant_on_layers(x, [20e-6, 2e-6], [200e9 / 0.7, 100e9 / 0.7])
        eps = piecewise_constant_on_layers(x, [20e-6, 2e-6], [0.0, -1e-3])
        result = compute_stress_and_ERR(x, M, eps, 20e-6)
        errors.append(abs(result["G"] - reference))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] / abs(reference) < 0.02


def test_coupled_model_is_bounded_and_relaxes(tmp_path):
    r = simulate(nx=221, dt=4.0, t_end=2000.0)
    assert r["c"].min() >= -1e-10 and r["c"].max() <= 1 + 1e-10
    assert np.min(r["G"]) >= -1e-10
    assert r["G"][-1] < 0.01 * r["G"].max()
    save_results(r, tmp_path, plots=False)
    assert (tmp_path / "profiles.csv").is_file() and (
        tmp_path / "summary.json"
    ).is_file()

"""Reproduce the coursework bilayer diffusion, mismatch stress and energy model."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from solvers.layered_transport import (
    solve_layered_diffusion,
    piecewise_constant_on_layers,
)
from solvers.coupled_stress import compute_stress_and_ERR

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "substrate_thickness_m": 20e-6,
    "film_thickness_m": 2e-6,
    "substrate_E_Pa": 200e9,
    "film_E_Pa": 100e9,
    "substrate_poisson": 0.30,
    "film_poisson": 0.30,
    "substrate_D_m2_s": 1e-12,
    "film_D_m2_s": 1e-10,
    "beta": 2e-3,
}


def simulate(nx: int = 441, dt: float = 2.0, t_end: float = 2000.0) -> dict:
    if nx < 45 or (nx - 1) % 11:
        raise ValueError(
            "nx must be at least 45 and (nx-1) divisible by 11 so the interface is on a grid node."
        )
    if t_end <= 0:
        raise ValueError("t_end must be positive.")
    p = DEFAULTS.copy()
    h = [p["substrate_thickness_m"], p["film_thickness_m"]]
    times = sorted(
        set(
            [
                0.0,
                *np.linspace(0.0, t_end, 121),
                *[t for t in [5.0, 200.0, 1600.0] if t <= t_end],
            ]
        )
    )
    transport = solve_layered_diffusion(
        layer_thicknesses=h,
        layer_diffusivities=[p["substrate_D_m2_s"], p["film_D_m2_s"]],
        nx=nx,
        t_end=t_end,
        dt=dt,
        c_initial=np.zeros(nx),
        left_bc=("neumann", 0.0),
        right_bc=("dirichlet", 1.0),
        save_times=times,
    )
    moduli = [
        p["substrate_E_Pa"] / (1 - p["substrate_poisson"]),
        p["film_E_Pa"] / (1 - p["film_poisson"]),
    ]
    M = piecewise_constant_on_layers(transport.x, h, moduli)
    mechanics = [
        compute_stress_and_ERR(transport.x, M, p["beta"] * c, h[0])
        for c in transport.concentrations
    ]
    G = np.array([m["G"] for m in mechanics])
    sigma = np.array([m["sigma"] for m in mechanics])
    peak = int(np.argmax(G))
    summary = {
        "scope": "Synthetic coursework model; not measured film properties or failure prediction",
        "nx": nx,
        "max_dt_s": dt,
        "t_end_s": t_end,
        "saved_times": len(times),
        "concentration_min": float(transport.concentrations.min()),
        "concentration_max": float(transport.concentrations.max()),
        "peak_G_J_m2": float(G[peak]),
        "peak_sample_time_s": float(transport.times[peak]),
        "final_G_J_m2": float(G[-1]),
        "final_max_abs_stress_Pa": float(np.abs(sigma[-1]).max()),
        "parameters": p,
    }
    return dict(
        x=transport.x,
        times=transport.times,
        c=transport.concentrations,
        sigma=sigma,
        G=G,
        summary=summary,
    )


def save_results(result: dict, output_dir: Path, *, plots: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "energy_release.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "energy_release_J_m2"])
        writer.writerows(zip(result["times"], result["G"]))
    indices = sorted(
        set(
            int(np.argmin(abs(result["times"] - t)))
            for t in [5.0, 200.0, 1600.0, result["times"][-1]]
            if t <= result["times"][-1]
        )
    )
    with (output_dir / "profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time_s", "x_m", "concentration", "mismatch_strain", "stress_Pa"]
        )
        for i in indices:
            writer.writerows(
                (
                    result["times"][i],
                    x,
                    c,
                    result["summary"]["parameters"]["beta"] * c,
                    s,
                )
                for x, c, s in zip(result["x"], result["c"][i], result["sigma"][i])
            )
    if not plots:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    for i in indices:
        label = f"{result['times'][i]:g} s"
        axes[0].plot(result["x"] * 1e6, result["c"][i], label=label)
        axes[1].plot(result["x"] * 1e6, result["sigma"][i] / 1e6, label=label)
    for ax in axes[:2]:
        ax.axvline(20, color=".5", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Distance from substrate bottom (µm)")
    axes[0].set(
        ylabel="Normalized concentration", title="Diffusion through the bilayer"
    )
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Biaxial stress (MPa)", title="Mismatch stress")
    axes[2].plot(result["times"], result["G"] * 1e3, color="#147e83")
    axes[2].set(
        xlabel="Time (s)",
        ylabel="Energy release rate (mJ/m²)",
        title="Delamination driving energy",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Transport-coupled stress in a bilayer", fontsize=16)
    fig.supxlabel(
        "Illustrative course model • linear elasticity and assumed material parameters",
        fontsize=9,
    )
    fig.savefig(output_dir / "overview.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=441)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--t-end", type=float, default=2000.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "demo")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = simulate(args.nx, args.dt, args.t_end)
    save_results(result, args.output_dir, plots=not args.no_plots)
    print(json.dumps(result["summary"], indent=2))
    print("Results:", args.output_dir)


if __name__ == "__main__":
    main()

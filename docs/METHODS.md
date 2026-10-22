# Methods, provenance and limitations

## Project origin

This repository adapts John Goodman's Spring 2026 UCSB MATRL 263 / ME 267 coursework: the final transport/stress driver and his HW1/HW2 solvers. John confirmed that he wrote the code himself or with AI assistance. The original numerical functions are retained in the solver modules, with a new importable driver and executable tests. This is a course project, not a publication or a lab-data release.

The course problem specified bilayer geometry, a surface concentration step, a sealed bottom and linear concentration-to-eigenstrain coupling. Public packaging excludes the assignment PDF and lecture material. Original course-specific file paths have been replaced by local module imports; the demo has no dependencies on other folders on the author's computer.

## Governing equations

Transport follows `dc/dt = d/dx(D dc/dx)` on a uniform mesh with piecewise-constant diffusivity. Backward Euler gives first-order temporal accuracy. Harmonic face diffusivities represent continuity of flux at a perfect interface.

Mismatch strain is `epsilon_star = beta c`. The biaxial modulus is `M = E/(1-nu)`. For a free laminate, stiffness and mismatch integrals give the 2x2 system for membrane strain `epsilon0` and curvature `kappa`. Biaxial stress is `sigma = M(epsilon0 + kappa x - epsilon_star)`.

For equibiaxial plane stress the elastic energy density is `M epsilon_elastic^2`. Delamination driving energy is `G = U_bonded - U_substrate_free - U_film_free`. It is an idealized energy difference between freely relaxed configurations, with no crack-front, mode-mix, interface-toughness or delamination-kinetics model.

The mechanics background follows [Begley and Hutchinson, The Mechanics and Reliability of Films, Multilayers and Coatings](https://www.cambridge.org/core/books/abs/mechanics-and-reliability-of-films-multilayers-and-coatings/frontmatter/C4D643D21AA4682886B197BCDD04C8C5), Cambridge University Press, 2017. That reference provides physical context; numerical tests verify the implementation independently through analytical limits rather than reproducing course solution files.

## Changes in the public edition

The original solver saved the first state after a requested snapshot while labeling it with the earlier request. For example, a 5 s request with a 2 s step could be labeled 5 s while containing the 6 s state. The public diffusion solver shortens a step to land on requested snapshot times. A uniform-source regression verifies `c=t` exactly at noninteger snapshot times.

Input checks now reject nonpositive or nonfinite time steps, invalid final times and snapshots outside the run interval. Stress functions are extracted from the original driver so tests can import them without running plots or writing files. The analytical laminate reference and thermal-conduction analogue are retained; the thermal analogue is outside the demonstrated/tested diffusion workflow.

## Interpretation

All material values and coupling coefficients are modeling assumptions. Linear isotropic elasticity, perfect bonding before idealized separation, one-dimensional transport and fixed material properties exclude plasticity, nonlinear swelling, reaction, creep, damage and evolving diffusion coefficients. The default 2 s maximum step does not resolve a 0.04 s film timescale. Mesh/time refinement and comparison with relevant measured properties are necessary before applying this model to a particular material system.

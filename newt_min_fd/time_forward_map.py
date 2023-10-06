import jax
import jax_cfd.base  as cfd
from typing import Callable, Tuple

def kolmogorov_ck13_step(
    dt: float,
    Re: float,
    grid: cfd.grids.Grid,
    wave_number: int=4
) -> Callable[[Tuple[cfd.grids.GridVariable]], Tuple[cfd.grids.GridVariable]]:
  """ Function to march a timestep """
  convect_fn = lambda v: tuple(cfd.advection.advect_linear(u, v, dt) for u in v)
  forcing_fn = cfd.forcings.kolmogorov_forcing(grid, k=wave_number)

  return cfd.equations.semi_implicit_navier_stokes(density = 1.,
                                                   viscosity = 1./Re,
                                                   dt = dt,
                                                   grid = grid,
                                                   forcing = forcing_fn,
                                                   convect = convect_fn)

def generate_time_forward_map(
    dt: float,
    Nt: int,
    grid: cfd.grids.Grid,
    Re: float
) -> Callable[[Tuple[cfd.grids.GridVariable]], Tuple[cfd.grids.GridVariable]]:

  step_fn = kolmogorov_ck13_step(dt, Re, grid)

  time_forward_map = cfd.funcutils.repeated(jax.remat(step_fn), steps = Nt)
  return jax.jit(time_forward_map)
""" Time forward maps: spectral coeffs -> spectral coeffs """
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral

from jax.config import config
config.update("jax_enable_x64", True)

from typing import Callable, Tuple, List, Union
Array = Union[np.ndarray, jnp.ndarray]

from collections import namedtuple
State = namedtuple("State", "steps T omega_old omega_new avg_observable")


def kolmogorov_ck13_step(viscosity, grid, smooth=True):
  wave_number = 4
  offsets = ((0, 0), (0, 0))
  # pylint: disable=g-long-lambda
  forcing_fn = lambda grid: cfd.forcings.kolmogorov_forcing(
      grid, k=wave_number, offsets=offsets)
  return spectral.equations.NavierStokes2D(
      viscosity,
      grid,
      drag=0.0,
      smooth=smooth,
      forcing_fn=forcing_fn)

def generate_time_forward_map(
    dt: float,
    Nt: int,
    grid: cfd.grids.Grid,
    viscosity: float
) -> Callable[[Array], Array]:
  
  step_fn = spectral.time_stepping.crank_nicolson_rk4(
    kolmogorov_ck13_step(viscosity, grid, smooth=True), dt)

  time_forward_map = cfd.funcutils.repeated(jax.remat(step_fn), steps=Nt)
  return jax.jit(time_forward_map)
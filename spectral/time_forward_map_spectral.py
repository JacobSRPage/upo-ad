""" Time forward maps: spectral coeffs -> spectral coeffs """
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import diffrax # TODO -- replace with corresponding equinox implementation 

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


def advance_velocity_module(step_fun, dt, obs_fn, max_steps=None):
  """Returns a function that takes State(time=t0) and returns a State(t0+T), 
      obs_fn(state.omega) -> float is an observable to be averaged along the trajectory.."""

  def cond_fun(state):
    """When this returns true, continue time stepping `state`."""
    return dt * state.steps < state.T

  def body_fun(state, _):
    """Increment `state` by one time step."""
    omega_new = step_fn(state.omega_new)
    observable = obs_fn(omega_new)
    obs_avg = (state.avg_observable * state.steps + observable) / (state.steps + 1)
    return State(steps=state.steps + 1,
                 T = state.T,
                 omega_old=state.omega_new,
                 omega_new=omega_new,
                 avg_observable = obs_avg)

  # Define a diffrax loop function, assigning every arg except init_val
  bounded_while_fun = partial(
      diffrax.misc.bounded_while_loop,
      cond_fun=cond_fun,
      body_fun=body_fun,
      max_steps=max_steps,
      base=64)

  def interpolate_fun(state):
    """Interpolate between v_old and v_new to get v at time=T."""
    time_old = (state.steps - 1) * dt
    time_new = state.steps * dt
    delta_t = time_new - time_old
    step_fraction = (state.T - time_old) / delta_t
    
    delta_omega = state.omega_new - state.omega_old
    omega_T = state.omega_old + delta_omega * step_fraction
    
    # Return State interpolated to the fractional step, where steps * dt = T
    return State(
        steps = state.steps + step_fraction,
        T = state.T,
        omega_old = state.omega_old,
        omega_new = omega_T,
        avg_observable = state.obs_avg)
  
  def time_advance_state_fun(state):
    state_final = bounded_while_fun(init_val=state)
    return interpolate_fun(state_final)

  return jax.jit(time_advance_state_fun)

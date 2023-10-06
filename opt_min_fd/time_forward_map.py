from typing import Callable, Tuple, List, Union

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
import equinox

from collections import namedtuple

Array = Union[np.ndarray, jnp.ndarray]
State = namedtuple("State", "steps T v_old v_new avg_observable")

import jax_cfd.base  as cfd

def advance_velocity_module(step_fun, dt, obs_fn, max_steps=None):
  """Returns a function that takes State(time=t0) and returns a State(t0+T)."""

  def cond_fun(state):
    """When this returns true, continue time stepping `state`."""
    return dt * state.steps < state.T

  def body_fun(state): 
    """Increment `state` by one time step."""
    v_update = step_fun(state.v_new)
    observable = obs_fn(v_update)
    obs_avg = (state.avg_observable * state.steps + observable) / (state.steps + 1)
    return State(steps=state.steps + 1,
                 T = state.T,
                 v_old=state.v_new,
                 v_new=v_update,
                 avg_observable=obs_avg)

  # Define a diffrax loop function, assigning every arg except init_val
  bounded_while_fun = partial(
      equinox.internal.while_loop,
      cond_fun=cond_fun,
      body_fun=body_fun,
      max_steps=max_steps,
      kind="bounded",
      base=64)

  def interpolate_fun(state):
    """Interpolate between v_old and v_new to get v at time=T."""
    time_old = (state.steps - 1) * dt
    time_new = state.steps * dt
    delta_t = time_new - time_old
    step_fraction = (state.T - time_old) / delta_t
    # Operations on GridArray
    delta_v_array = tuple(u_new.array - u_old.array
                          for u_old, u_new in zip(state.v_old, state.v_new))
    v_array = tuple(u_old.array + delta_u_array * step_fraction
                    for u_old, delta_u_array in zip(state.v_old, delta_v_array))
    # Cast to Grid Variable (associate BCs)
    v_T = tuple(cfd.grids.GridVariable(u_array, u.bc)
                for u_array, u in zip(v_array, state.v_old))
    # Return State interpolated to the fractional step, where steps * dt = T
    # TODO update avg observable (will need to time integrate...)
    return State(
        steps = state.steps + step_fraction,
        T = state.T,
        v_old = state.v_old,
        v_new = v_T,
        avg_observable = state.avg_observable)
  
  def time_advance_state_fun(state):
    state_final = bounded_while_fun(init_val=state)
    return interpolate_fun(state_final)

  return jax.jit(time_advance_state_fun)

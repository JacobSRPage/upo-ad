""" Loss functions for searching for periodic orbits with jax-cfd """ 
import typing
from typing import Callable, Tuple, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

Array = Union[np.ndarray, jnp.ndarray]

import jax_cfd.base  as cfd
import interact_jaxcfd_dtypes as glue
from time_forward_map import State

def loss_fn_diffrax(
    u0: Tuple[cfd.grids.GridArray], 
    T: float, 
    x_shift: float,
    forward_map: Callable[[State], State]
) -> float:
  """ 
    Credit to P.N. for idea + help on implementation 
  """
  # TODO move x_shift into state 
  state_0 = State(steps = 0,
                T = T,
                v_old = u0,
                v_new = u0,
                avg_observable = 0.) 
  state_T = forward_map(state_0) 

  u_T = state_T.v_new
  shifted_uT = glue.x_shift_field(u_T, x_shift)
  loss = jnp.linalg.norm(glue.state_vector(shifted_uT) - glue.state_vector(u0), ord=2) / jnp.linalg.norm(glue.state_vector(u0), ord=2)
  return loss

def loss_fn_diffrax_target_obs(
    u0: Tuple[cfd.grids.GridArray], 
    T: float, 
    x_shift: float,
    forward_map: Callable[[State], State],
    obs_target: float
) -> float:
  """ 
    Credit to P.N. for idea + help on implementation 
  """
  # TODO move x_shift into state 
  state_0 = State(steps = 0,
                T = T,
                v_old = u0,
                v_new = u0,
                avg_observable = 0. #obs_init 
                ) 
  state_T = forward_map(state_0) 

  u_T = state_T.v_new
  shifted_uT = glue.x_shift_field(u_T, x_shift)
  loss = jnp.linalg.norm(glue.state_vector(shifted_uT) - glue.state_vector(u0), ord=2) / jnp.linalg.norm(glue.state_vector(u0), ord=2) + \
        1. * jax.nn.sigmoid(100 * (obs_target - state_T.avg_observable)) #- (state.avg_observable - obs_target) ** 2
  return loss

def loss_fn_diffrax_targetT(
    u0: Tuple[cfd.grids.GridArray], 
    T: float, 
    x_shift: float,
    T_target: float,
    forward_map: Callable[[State], State]
) -> float:
  """ 
    Credit to P.N. for idea + help on implementation 
  """
  # TODO move x_shift into state 
  state_0 = State(steps = 0,
                T = T,
                v_old = u0,
                v_new = u0,
                avg_observable = 0.) 
  state_T = forward_map(state_0) 

  u_T = state_T.v_new
  shifted_uT = glue.x_shift_field(u_T, x_shift)
  loss = jnp.linalg.norm(glue.state_vector(shifted_uT) - glue.state_vector(u0), ord=2) / jnp.linalg.norm(glue.state_vector(u0), ord=2) + 0.01 * (T - T_target) ** 2
  return loss


def loss_fn_diffrax_nomean(
    u0: Tuple[cfd.grids.GridVariable],
    T: float,
    x_shift: float,
    forward_map: Callable[[State], State]
) -> float:
  state_0 = State(steps = 0,
                T = T,
                v_old = u0,
                v_new = u0,
                avg_observable = 0.) 
  state_T = forward_map(state_0) 

  u_T = state_T.v_new
  shifted_uT = glue.x_shift_field(u_T, x_shift)

  U_mean, V_mean = glue.mean_flows(u0)
  loss = jnp.linalg.norm(glue.state_vector(shifted_uT) - glue.state_vector(u0), ord=2) / jnp.linalg.norm(glue.state_vector(u0), ord=2) + 1000. * (V_mean ** 2)
  return loss
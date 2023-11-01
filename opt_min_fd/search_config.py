""" Basic hacky wrappers for optimisation + damping V0 """
from functools import partial
import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers as optimizers

import numpy as np
import jax_cfd.base as cfd 
import time_forward_map as tfm
import interact_jaxcfd_dtypes as glue
import loss_functions as lf
import optimisation as op

from typing import Tuple, Callable

class KolFlowSimulationConfig:
  def __init__(self, Re: float, grid: cfd.grids.Grid, T_burn: float=50.):
    self.Re = Re 
    self.grid = grid 
    self.wavenumber = 4
    velocity_est = self.Re / self.wavenumber ** 2
    self.dt_stable = cfd.equations.stable_time_step(velocity_est, 0.5, 1. / self.Re, self.grid) / 2.

    kolmogorov = cfd.forcings.kolmogorov_forcing(self.grid, k=self.wavenumber)
    self.step_fn = cfd.equations.semi_implicit_navier_stokes(
      density=1.,
      viscosity=1./self.Re,
      dt=self.dt_stable,
      grid=self.grid,
      convect=lambda v: tuple(cfd.advection.advect_linear(u, v, self.dt_stable) for u in v),
      forcing=kolmogorov
    )

    self.T_burn = T_burn
    self.time_forward_map_burn = tfm.advance_velocity_module(self.step_fn,
                                                             self.dt_stable,
                                                             lambda x: 0.,
                                                             max_steps=int(self.T_burn / self.dt_stable)+1)

  def generate_random_ic(self, seed=None):
    if seed is None:
      seed = np.random.randint(1e4)
    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(seed), self.grid, 5.)
    state_0 = tfm.State(steps = 0,
                        T = self.T_burn,
                        v_old = v0,
                        v_new = v0,
                        avg_observable = 0.)
    state_T_burn = self.time_forward_map_burn(state_0)
    return state_T_burn.v_new


class PeriodicSearchConfig:  
  def __init__(self, 
               flow_config: KolFlowSimulationConfig,
               T_guess: float,
               N_opt: int,
               N_opt_damp: int,
               loss_thresh: float
               ):
    self.flow = flow_config
    self.T_guess = T_guess 
    self.N_opt = N_opt
    self.N_opt_damp = N_opt_damp
    self.thresh = loss_thresh
    self.learning_rate = 0.35
    self.learning_rate_damp_v = 0.1
    self.n_shift_reflects = 0 # universal 

    advance_velocity_fn = tfm.advance_velocity_module(self.flow.step_fn,
                                                      self.flow.dt_stable,
                                                      lambda x: 0.,
                                                      max_steps=2 * int(self.T_guess / 
                                                                        self.flow.dt_stable))
    self.loss_fn = partial(lf.loss_fn_diffrax, forward_map=advance_velocity_fn)
    self.grad_u_T_shift = partial(glue.grad_u_with_extras_vec, loss_fn=self.loss_fn)
    self.optimiser_triple = optimizers.adagrad(self.learning_rate)

    self.loss_fn_damp_v = partial(lf.loss_fn_diffrax_nomean, forward_map=advance_velocity_fn)
    self.grad_u_T_shift_damp_v = partial(glue.grad_u_with_extras_vec, loss_fn=self.loss_fn_damp_v)
    self.optimiser_triple_damp_v = optimizers.adagrad(self.learning_rate_damp_v)

  def initial_search(self):
    v_initial = self.flow.generate_random_ic()
    initial_ar, offsets = glue.gv_tuple_to_jnp(v_initial)
    initial_shape = initial_ar.shape
    bc = v_initial[0].bc

    performance, u_T_shift = op.iterate_optimizer_for_rpo(self.optimiser_triple, 
                                                          v_initial, self.flow.grid, 
                                                          self.loss_fn, self.grad_u_T_shift, 
                                                          self.T_guess, 0., self.N_opt)


    T_out = u_T_shift[-2]
    shift_out = u_T_shift[-1]
    u0_out = glue.jnp_to_gv_tuple(u_T_shift[:-2].reshape(initial_shape), offsets, self.flow.grid, bc)
    return u0_out, T_out, shift_out, performance[0][-1]
  
  def damp_v(self, u0, T0, shift0):
    initial_ar, offsets = glue.gv_tuple_to_jnp(u0)
    initial_shape = initial_ar.shape
    bc = u0[0].bc
    
    performance, u_T_shift = op.iterate_optimizer_for_rpo(self.optimiser_triple_damp_v, 
                                                          u0, self.flow.grid, 
                                                          self.loss_fn_damp_v, self.grad_u_T_shift_damp_v, 
                                                          T0, shift0, self.N_opt_damp)
    T_newton = u_T_shift[-2]
    shift_newton = u_T_shift[-1]
    u0_newton = glue.jnp_to_gv_tuple(u_T_shift[:-2].reshape(initial_shape), offsets, self.flow.grid, bc)
    return u0_newton, T_newton, shift_newton, performance[0][-1]

  def loop_opt(self, n_guess_max=None):
    n_guess = 0
    n_success = 0
    while True:
      if n_guess_max is not None:
        if n_guess > n_guess_max:
          break
      print("This is guess number ", n_guess+1)
      u0, T0, shift0, loss_0 = self.initial_search()
      if (loss_0 < self.thresh) and (T0 > self.flow.dt_stable):
        print("Promising first run, damping V0 ... ")
        u, T, shift, loss = self.damp_v(u0, T0, shift0)
        if (loss < self.thresh) and (T > self.flow.dt_stable):
          n_success += 1
          umean, _ = glue.mean_flows(u)
          shift_update = shift - umean * T
          file_front = ('guesses_with_damp_Re' + str(self.flow.Re) + 
                        '_T' + str(self.T_guess) + 
                        '_Nopt' + str(self.N_opt) + 
                        '_Noptdamp' + str(self.N_opt_damp) +
                        '_thresh' + str(self.thresh) + 
                        '_file' + str(n_success))
          
          print("Success! Good guess number ", n_success)
          write_guess_and_metadata(u, T, shift_update, self.n_shift_reflects, loss, file_front)
      n_guess += 1

class PeriodicSearchShiftReflectConfig(PeriodicSearchConfig):  
  def __init__(self, 
               flow_config: KolFlowSimulationConfig,
               T_guess: float,
               n_shift_reflects: int,
               N_opt: int,
               N_opt_damp: int,
               loss_thresh: float
               ):
    self.flow = flow_config
    self.T_guess = T_guess 
    self.N_opt = N_opt
    self.N_opt_damp = N_opt_damp
    self.thresh = loss_thresh
    self.learning_rate = 0.35
    self.learning_rate_damp_v = 0.1
    self.n_shift_reflects = n_shift_reflects

    # setup shift reflect function
    shift_reflect_fn = partial(glue.shift_reflect_field, n_shift_reflects=self.n_shift_reflects, n_waves=4)
    shift_reflect_fn = jax.jit(shift_reflect_fn)

    advance_velocity_fn = tfm.advance_velocity_module(self.flow.step_fn,
                                                      self.flow.dt_stable,
                                                      lambda x: 0.,
                                                      max_steps=2 * int(self.T_guess / 
                                                                        self.flow.dt_stable))
    self.loss_fn = partial(lf.loss_fn_diffrax_shift_reflects, forward_map=advance_velocity_fn, shift_reflect_fn=shift_reflect_fn)
    self.grad_u_T_shift = partial(glue.grad_u_with_extras_vec, loss_fn=self.loss_fn)
    self.optimiser_triple = optimizers.adagrad(self.learning_rate)

    self.loss_fn_damp_v = partial(lf.loss_fn_diffrax_nomean_shift_reflect, forward_map=advance_velocity_fn, shift_reflect_fn=shift_reflect_fn)
    self.grad_u_T_shift_damp_v = partial(glue.grad_u_with_extras_vec, loss_fn=self.loss_fn_damp_v)
    self.optimiser_triple_damp_v = optimizers.adagrad(self.learning_rate_damp_v)

class TargetedSearchConfig(PeriodicSearchConfig):
  """ Time-average observable above some threshold """
  def __init__(self, 
               flow_config: KolFlowSimulationConfig,
               T_guess: float,
               N_opt: int,
               N_opt_damp: int,
               loss_thresh: float,
               observable_fn: Callable[[Tuple[cfd.grids.GridVariable]], float]=lambda x: 0.,
               observable_thresh: float=0.,
               greater_than_target: int=1
               ):
    self.flow = flow_config
    self.T_guess = T_guess 
    self.N_opt = N_opt
    self.N_opt_damp = N_opt_damp
    self.thresh = loss_thresh
    self.learning_rate = 0.35
    self.learning_rate_damp_v = 0.1
    self.observable_fn = observable_fn
    self.observable_thresh = observable_thresh
    self.greater_than = greater_than_target

    advance_velocity_fn = tfm.advance_velocity_module(self.flow.step_fn,
                                                      self.flow.dt_stable,
                                                      self.observable_fn,
                                                      max_steps=2 * int(self.T_guess / 
                                                                        self.flow.dt_stable))
    self.loss_fn = partial(
      lf.loss_fn_diffrax_target_obs, 
      forward_map=advance_velocity_fn,
      obs_target=self.observable_thresh,
      greater_than_target=self.greater_than
      )
    
    self.grad_u_T_shift = partial(glue.grad_u_with_extras_vec, loss_fn=self.loss_fn)
    self.optimiser_triple = optimizers.adagrad(self.learning_rate)

    self.loss_fn_damp_v = partial(lf.loss_fn_diffrax_nomean, forward_map=advance_velocity_fn)
    self.grad_u_T_shift_damp_v = partial(glue.grad_u_with_extras_vec, loss_fn=self.loss_fn_damp_v)
    self.optimiser_triple_damp_v = optimizers.adagrad(self.learning_rate_damp_v)

  def initial_search(self):
    # ensure initial snapshot is in region of interest
    if self.greater_than == 1:
      observed_quantity = 0.
      while observed_quantity < self.observable_thresh:
        v_initial = self.flow.generate_random_ic()
        observed_quantity = self.observable_fn(v_initial)
    else:
      observed_quantity = 100.
      while observed_quantity > self.observable_thresh:
        v_initial = self.flow.generate_random_ic()
        observed_quantity = self.observable_fn(v_initial)

    initial_ar, offsets = glue.gv_tuple_to_jnp(v_initial)
    initial_shape = initial_ar.shape
    bc = v_initial[0].bc

    performance, u_T_shift = op.iterate_optimizer_for_rpo(self.optimiser_triple, 
                                                          v_initial, self.flow.grid, 
                                                          self.loss_fn, self.grad_u_T_shift, 
                                                          self.T_guess, 0., self.N_opt)


    T_out = u_T_shift[-2]
    shift_out = u_T_shift[-1]
    u0_out = glue.jnp_to_gv_tuple(u_T_shift[:-2].reshape(initial_shape), offsets, self.flow.grid, bc)
    return u0_out, T_out, shift_out, performance[0][-1]


def write_guess_and_metadata(u: Tuple[cfd.grids.GridVariable], 
                             T: float, shift: float, shift_reflects: int, loss: float,
                             file_front: str):
  meta_data = jnp.array([T, shift, shift_reflects, loss])
  u_ar, _ = glue.gv_tuple_to_jnp(u)
  np.save(file_front + '_meta.npy', meta_data)
  np.save(file_front + '_array.npy', u_ar)

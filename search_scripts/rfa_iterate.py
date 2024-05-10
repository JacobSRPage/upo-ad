""" Simple top level script to load in an IC and iterate.
    Will be embedded in search config in due course, this is 
    just proof of concept. """
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax_cfd.base as cfd
from functools import partial
import jax.example_libraries.optimizers as optimizers

import opt_newt_jaxcfd.interact_jaxcfd.time_forward_map as tfm
import opt_newt_jaxcfd.interact_jaxcfd.interact_jaxcfd_dtypes as glue
import opt_newt_jaxcfd.optimisation.loss_functions as lf
import opt_newt_jaxcfd.optimisation.optimisation as op

# guess file
u_v_jnp_guess = np.load('UPO_guess_RFAtest_0.npy')
meta_guess = np.load('UPO_guess_RFAtest_0_meta.npy')

# KF config
Re = 40. 
Nx = 256
Ny = 256
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

# define the guess 
T_guess, shift, n_shift_ref = meta_guess
n_shift_ref = int(n_shift_ref)

# convert the velocity array to a grid variable
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
offsets = [(1., 0.5), (0.5, 1.)]
bc = cfd.boundaries.HomogeneousBoundaryConditions(types=(('periodic', 'periodic'), 
                                                         ('periodic', 'periodic')))
# initial condition may not be ordered correctly
if u_v_jnp_guess.shape[0] == 2:
  u_v_jnp_guess = u_v_jnp_guess.transpose((1, 2, 0))
u_gv_guess = glue.jnp_to_gv_tuple(u_v_jnp_guess, offsets, grid, bc)
velocity_est = Re / 4 ** 2
dt_stable = cfd.equations.stable_time_step(velocity_est, 0.5, 1. / Re, grid) / 2.

# optimiser config
n_opt_steps = 500
n_damp_steps = 100
learning_rate = 0.35
learning_rate_damp_v = 0.1
loss_thresh = 0.02 # when to save a guess for Newton

# setup simulator
kolmogorov = cfd.forcings.kolmogorov_forcing(grid, k=4)
step_fn = cfd.equations.semi_implicit_navier_stokes(
  density=1.,
  viscosity=1. / Re,
  dt=dt_stable,
  grid=grid,
  convect=lambda v: tuple(cfd.advection.advect_linear(u, v, dt_stable) for u in v),
  forcing=kolmogorov
)
advance_velocity_fn = tfm.advance_velocity_module(step_fn,
                                                  dt_stable,
                                                  lambda x: 0.,
                                                  max_steps=2 * int(T_guess / 
                                                                    dt_stable))
# setup shift reflect function
shift_reflect_fn = partial(glue.shift_reflect_field, n_shift_reflects=n_shift_ref, n_waves=4)
shift_reflect_fn = jax.jit(shift_reflect_fn)

# setup loss functions
loss_fn = partial(lf.loss_fn_diffrax_shift_reflects, 
                  forward_map=advance_velocity_fn,
                  shift_reflect_fn=shift_reflect_fn)
grad_u_T_shift = partial(glue.grad_u_with_extras_vec, loss_fn=loss_fn)
optimiser_triple = optimizers.adagrad(learning_rate)

loss_fn_damp_v = partial(lf.loss_fn_diffrax_nomean_shift_reflect, 
                         forward_map=advance_velocity_fn, 
                         shift_reflect_fn=shift_reflect_fn)
grad_u_T_shift_damp_v = partial(glue.grad_u_with_extras_vec, loss_fn=loss_fn_damp_v)
optimiser_triple_damp_v = optimizers.adagrad(learning_rate_damp_v)

# now iterate 
initial_ar, offsets = glue.gv_tuple_to_jnp(u_gv_guess)
initial_shape = initial_ar.shape
bc = u_gv_guess[0].bc

performance, u_T_shift = op.iterate_optimizer_for_rpo(optimiser_triple, 
                                                      u_gv_guess, grid, 
                                                      loss_fn, grad_u_T_shift, 
                                                      T_guess, shift, n_opt_steps)


T_out = u_T_shift[-2]
shift_out = u_T_shift[-1]
u0_out = glue.jnp_to_gv_tuple(u_T_shift[:-2].reshape(initial_shape), offsets, grid, bc)

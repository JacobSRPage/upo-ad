from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax_cfd.base as cfd

from opt_newt_jaxcfd.optimisation import search_config

# KF config
T_guess = 5.
Re = 40. 
Nx = 256
Ny = 256
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

n_shift_reflects = 3

# optimiser config
n_opt_steps = 500
n_damp_steps = 200
loss_thresh = .02 # when to save a guess for Newton


grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
flow_setup = search_config.KolFlowSimulationConfig(Re, grid)
opt_setup = search_config.PeriodicSearchShiftReflectConfig(flow_setup, T_guess, n_shift_reflects, 
                                                           n_opt_steps, n_damp_steps, loss_thresh)

opt_setup.loop_opt()

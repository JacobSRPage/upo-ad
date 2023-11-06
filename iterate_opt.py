from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax_cfd.base as cfd

from opt_newt_jaxcfd.optimisation import search_config

# KF config
T_guess = 5.
Re = 40. 
Nx = 128
Ny = 128
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

# optimiser config
n_opt_steps = 20
n_damp_steps = 10
loss_thresh = 2. # when to save a guess for Newton

grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
flow_setup = search_config.KolFlowSimulationConfig(Re, grid)
opt_setup = search_config.PeriodicSearchConfig(flow_setup, T_guess,
                                               n_opt_steps, n_damp_steps,
                                               loss_thresh)

opt_setup.loop_opt()

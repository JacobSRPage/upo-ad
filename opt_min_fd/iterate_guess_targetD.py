from jax.config import config
config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
import jax_cfd.base as cfd
import search_config
import diagnostics as dg

# KF config
T_guess = 5.5
Re = 40.
Nx = 256
Ny = 256
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

diss_threshold = 0.1
greater_than = 0

# optimiser config
n_opt_steps = 250
n_damp_steps = 100
loss_thresh = 0.5 # when to save a guess for Newton

grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
flow_setup = search_config.KolFlowSimulationConfig(Re, grid)

# configure specific search
diss_fn = partial(dg.compute_diss_snapshot, grid=grid, Re=Re, n_kol_waves=4)
opt_setup = search_config.TargetedSearchConfig(flow_setup, 
                                               T_guess,
                                               n_opt_steps, 
                                               n_damp_steps,
                                               loss_thresh,
                                               observable_fn=diss_fn,
                                               observable_thresh=diss_threshold,
                                               greater_than_target=greater_than
                                               )

opt_setup.loop_opt()

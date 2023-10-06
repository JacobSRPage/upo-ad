from jax.config import config
config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
import jax_cfd.base as cfd
import search_config
import diagnostics as dg

# KF config
T_guess = 5.
Re = 40. 
Nx = 128
Ny = 128
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi
energy_threshold = 0.45

# optimiser config
n_opt_steps = 20
n_damp_steps = 10
loss_thresh = 2. # when to save a guess for Newton

grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
flow_setup = search_config.KolFlowSimulationConfig(Re, grid)

# configure specific search
energy_fn = partial(dg.compute_energy_snapshot(grid=grid, Re=Re, n_kol_waves=4))
opt_setup = search_config.TargetedSearchConfig(flow_setup, 
                                               T_guess,
                                               n_opt_steps, 
                                               n_damp_steps,
                                               loss_thresh,
                                               observable_fn=energy_fn,
                                               observable_thresh=energy_threshold
                                               )

opt_setup.loop_opt()

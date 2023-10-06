from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import pickle
from jax.config import config
config.update("jax_enable_x64", True)
import jax_cfd.base as cfd

import time_forward_map as tfm
import newton as nt
import interact_jaxcfd_dtypes as glue

guess_files= ['guesses_Jun5/run0/guesses_with_damp_Re100_T2.5_Nopt250_Noptdamp100_thresh0.02_file' + 
              str(j) + '.obj' for j in range(4)]
guess_files += ['guesses_Jun5/run1/guesses_with_damp_Re100_T2.75_Nopt275_Noptdamp125_thresh0.02_file' + 
                str(j) + '.obj' for j in range(8)]
guess_files += ['guesses_Jun5/run1/guesses_with_damp_Re100_T3.25_Nopt325_Noptdamp150_thresh0.02_file' +
                str(j) + '.obj' for j in range(3)]

success_file_name = 'converged_Jun5_Re100.obj'

# DNS configuration should match guesses! 
Nx = 512
Ny = 512
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

Re = 100.
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))

max_velocity = 5. # estimate (not prescribed)
dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, 1./Re, grid) / 2. # should take care here

newton_solver = nt.rpoSolver(dt_stable, grid, Re, nmax_newt=30, nmax_gm=50, nmax_hook=8)

successes = {}
count = 0
for guess_file_name in guess_files:
  guess_file = open(guess_file_name, 'rb') 
  guesses = pickle.load(guess_file)
  guess_file.close()

  for guess_num in guesses:
    guess = guesses[guess_num]
    um, vm = glue.mean_flows(guess.u_init)
    guess.shift_init += um * guess.T_init # assume we have subtracted off in the generation
    newt_result = newton_solver.iterate(guess)
    if newt_result.newt_resi_history[-1] < newton_solver.eps_newt:
      successes[count] = newt_result
      count += 1
    
    with open(success_file_name, 'wb') as f:
      pickle.dump(successes, f)

from functools import partial
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from jax.config import config
config.update("jax_enable_x64", True)
import jax_cfd.base as cfd

import time_forward_map as tfm
import newton as nt
import interact_jaxcfd_dtypes as glue

file_front = 'guesses_Re40/guesses_with_damp_Re40.0_T4.0_Nopt400_Noptdamp100_thresh0.05_file'
N_files = 10
array_files = [file_front + str(j) + '_array.npy' for j in range(N_files)]
meta_files = [file_front + str(j) + '_meta.npy' for j in range(N_files)]

# DNS configuration should match guesses! 
Nx = 512
Ny = 512
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi
add_mean_flow = True # set to True if mean flow (in x) was subtracted when estimating shift

Re = 100.
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))

max_velocity = 5. # estimate (not prescribed)
dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, 1./Re, grid) / 2. # should take care here

newton_solver = nt.rpoSolver(dt_stable, grid, Re, nmax_newt=30, nmax_gm=50, nmax_hook=8)

# configure variables for setting up GridVariable 
offsets = [(1., 0.5), (0.5, 1.)]
bc = cfd.boundaries.HomogeneousBoundaryConditions(types=(('periodic', 'periodic'), ('periodic', 'periodic')))

count = 0
for ar_file, meta_file in zip(array_files, meta_files):
  # construct GridVariable from input arrays
  guess_ar = np.load(ar_file)
  meta_ar = np.load(meta_file)
  T = meta_ar[0]
  shift = meta_ar[1]

  # catch earlier guesses without shift reflects
  try:
    n_shift_reflects = meta_ar[2]
    if type(n_shift_reflects) != int:
      n_shift_reflects = 0
  except:
    n_shift_reflects = 0

  u_guess_gv = glue.jnp_to_gv_tuple(guess_ar, offsets, grid, bc) 
  guess = nt.poGuess(u_guess_gv, T, shift, n_shift_reflects=n_shift_reflects)
    
  # compute mean flow
  if add_mean_flow: 
    um, vm = glue.mean_flows(guess.u_init)
    guess.shift_init += um * guess.T_init # assume we have subtracted off in the generation
  newt_result = newton_solver.iterate(guess)
  if newt_result.newt_resi_history[-1] < newton_solver.eps_newt:
    ar_success, _ = glue.gv_tuple_to_jnp(guess.u_out)
    T_success = guess.T_out
    shift_success = guess.shift_out
    final_loss = guess.newt_resi_history[-1]
    try:
      np.save('success_' + str(count) + '_array.npy', ar_success)
      np.save('success_' + str(count) + '_meta.npy', np.array([T_success, 
                                                               shift_success, 
                                                               guess.n_shift_reflects,
                                                               final_loss]))
      count += 1
    except:
      print("Erroneous convergence. Moving on.")

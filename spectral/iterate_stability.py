""" Compute stability of RPOs -- Floquet multipliers and eigenvectors.
    Returns a dictionary, as before, keys: just an index. Vals [RPO, (eig, eigVec)] """  
import numpy as np
import pickle

import jax
import jax.numpy as jnp

import jax_cfd.base  as cfd
import jax_cfd.spectral as spectral

import newton_spectral as nt_sp
import interact_spectral as insp
import stability as st

from jax.config import config
config.update("jax_enable_x64", True)

rpo_file = 'Re100_convs/Re100_unified_05Dec.obj'
save_file = 'Re100_unified_05Dec_stability.obj'

# problem configuration (should match RPOs)
Nx = 512
Ny = 512
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

Re = 100.
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
max_velocity = 5.
dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, 1./Re, grid)

# load in solutions
with open(rpo_file, 'rb') as f:
  converged_solns = pickle.load(f) # keys should just be index 0, 1, 2... 

solns_and_eigs = {}
for indx in converged_solns:
  rpo = converged_solns[indx]
  print("#######")
  print("Computing stability of solution with T=", rpo.T_out, "and shift", rpo.shift_out)

  floq, floq_eigV = st.compute_rpo_stability(rpo, grid, dt_stable, Re, eps_fd=1e-7, N_eig=50)

  idx = np.abs(floq).argsort()[::-1]
  floq = floq[idx]
  floq_eigV = floq_eigV[:,idx]
  
  print("Most unstable exponent =", np.log(floq[0]) / rpo.T_out)
  print("#######")
  solns_and_eigs[indx] = [rpo, (floq, floq_eigV)]

with open(save_file, 'wb') as f:
  pickle.dump(solns_and_eigs, f)

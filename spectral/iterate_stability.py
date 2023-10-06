""" Compute stability of RPOs -- Floquet multipliers and eigenvectors.
    Returns a dictionary, as before, keys: just an index. Vals [RPO, (eig, eigVec)] """  
import numpy as np
import jax.numpy as jnp

import jax_cfd.base  as cfd

import newton_spectral as nt_sp
import stability as st

from jax.config import config
config.update("jax_enable_x64", True)

n_orbits = 45
start_orbit = 0

# problem configuration (should match RPOs)
Nx = 256
Ny = 256
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

Re = 40.
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
max_velocity = 2. * Re / (4 ** 2) # 2 * laminar soln 
dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, 1./Re, grid)

for n in range(start_orbit, start_orbit + n_orbits):
  # files (array and metadata) to be read in, and name of stab file to write out
  rpo_ar_file = 'soln_array_Re' + str(int(Re)) + '_' + str(n) + '.npy'
  meta_file = 'soln_meta_Re' + str(int(Re)) + '_' + str(n) + '.npy'
  stab_file = 'soln_eigs_Re' + str(int(Re)) + '_' + str(n) + '.npy'

  # create initial condition
  vort_rft = jnp.load(rpo_ar_file)
  meta = jnp.load(meta_file)
  
  rpo = nt_sp.poGuessSpectral(None, None, None, None, None)
  rpo.omega_rft_out = vort_rft
  rpo.T_out = meta[0]
  rpo.shift_out = meta[1]

  print("#######")
  print("Computing stability of solution with T=", rpo.T_out, "and shift", rpo.shift_out)
  floq, _ = st.compute_rpo_stability(rpo, grid, dt_stable, Re, eps_fd=1e-7, N_eig=50)

  idx = np.abs(floq).argsort()[::-1]
  floq = floq[idx]
  # floq_eigV = floq_eigV[:,idx]
  
  print("Most unstable exponent =", np.log(floq[0]) / rpo.T_out)
  print("#######")
  jnp.save(stab_file, floq)
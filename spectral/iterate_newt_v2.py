""" TODO -- move time advancement to specfic spectral routine. """
import jax.numpy as jnp
import numpy as np
import jax_cfd.base as cfd

from jax.config import config
config.update("jax_enable_x64", True)

import interact_jaxcfd_dtypes as glue
import newton_spectral as nt_sp

file_front = 'guesses_Jun5/guess_'

file_range = range(95)
array_files = [file_front + 'ar_' + str(j) + '.npy' for j in file_range]
meta_files = [file_front + 'meta_' + str(j) + '.npy' for j in file_range]

Nx = 512
Ny = 512
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

n_waves = 4 # number of forcing wavelengths
Re = 100.
viscosity = 1. / Re
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
max_velocity = 5. # estimate (not prescribed)
dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, viscosity, grid) 

newton_solver = nt_sp.rpoSolverSpectral(dt_stable, grid, Re, nmax_hook=10, eps_newt=1e-10)

#with open(guess_file, 'rb') as f:
#  guesses = pickle.load(f)

count = 0
#for guess_num in guesses:
for file_ar, file_meta in zip(array_files, meta_files):
  # convert finite difference solution with finite zero Fourier mode vels
  # to spectral representation
  v0 = glue.files_to_gv_tuple(file_ar, grid)
  T, shift, _ = np.load(file_meta)
  umean, _ = glue.mean_flows(v0)
  shift_update = shift - umean * T
  vorticity0 = cfd.finite_differences.curl_2d(v0).data
  omega0 = jnp.fft.rfftn(vorticity0)

  po_guess_spectral = nt_sp.poGuessSpectral(omega0, T, shift_update)
  
  po_update = newton_solver.iterate(po_guess_spectral)
  if po_update.newt_resi_history[-1] < newton_solver.eps_newt:
    np.save(file_front + str(count) + '_spec_array.npy', po_update.omega_rft_out)
    np.save(file_front + str(count) + '_spec_meta.npy', np.array([po_update.T_out, 
                                                                  po_update.shift_out,
                                                                  po_update.newt_resi_history[-1]]))
    count += 1

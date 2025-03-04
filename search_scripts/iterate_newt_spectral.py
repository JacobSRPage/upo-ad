""" TODO -- move time advancement to specfic spectral routine. """
import jax.numpy as jnp
import numpy as np
import jax_cfd.base as cfd

from jax.config import config
config.update("jax_enable_x64", True)

from opt_newt_jaxcfd.interact_jaxcfd import interact_jaxcfd_dtypes as glue
from opt_newt_jaxcfd.newton import newton_spectral as nt_sp

file_front = 'success_run5_'

file_range = range(2)
array_files = [file_front + str(j) + '_array.npy' for j in file_range]
meta_files = [file_front + str(j) + '_meta.npy' for j in file_range]

Nx = 256
Ny = 256
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

n_waves = 4 # number of forcing wavelengths
Re = 40.
viscosity = 1. / Re
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
max_velocity = 5. # estimate (not prescribed)
dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, viscosity, grid) 

newton_solver = nt_sp.rpoSolverSpectral(grid, nmax_hook=10, eps_newt=1e-10)

#with open(guess_file, 'rb') as f:
#  guesses = pickle.load(f)

count = 0
#for guess_num in guesses:
for file_ar, file_meta in zip(array_files, meta_files):
  # convert finite difference solution with finite zero Fourier mode vels
  # to spectral representation
  v0 = glue.files_to_gv_tuple(file_ar, grid)
  T, shift, n_sr, _ = np.load(file_meta)
  umean, _ = glue.mean_flows(v0)
  shift_update = shift - umean * T
  vorticity0 = cfd.finite_differences.curl_2d(v0).data
  omega0 = jnp.fft.rfftn(vorticity0)

  po_guess_spectral = nt_sp.poGuessSpectral(omega0, T, shift_update, n_sr)
  
  po_update = newton_solver.iterate(po_guess_spectral, Re, dt_stable)
  if po_update.newt_resi_history[-1] < newton_solver.eps_newt:
    np.save(file_front + str(count) + '_spec_array.npy', po_update.omega_rft_out)
    np.save(file_front + str(count) + '_spec_meta.npy', np.array([po_update.T_out, 
                                                                  po_update.shift_out,
                                                                  po_update.n_shift_reflects,
                                                                  Re]))
    count += 1

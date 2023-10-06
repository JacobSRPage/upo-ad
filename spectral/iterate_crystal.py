""" TODO -- move time advancement to specfic spectral routine. """
from functools import partial

import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

import pickle
import matplotlib.pyplot as plt

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral

from jax.config import config
config.update("jax_enable_x64", True)

import interact_jaxcfd_dtypes as glue
import newton_spectral as nt_sp

file_front = 'guesses_Jun5/guess_'
array_files = ['Re100_crystal_l2/Re100_l2_guesses_' + str(n) + '.npy' for n in range(10)]

T_burst = 2.
shift_burst = 0. # set this uniformly over the data

Nx = 128
Ny = 128
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

n_waves = 4 # number of forcing wavelengths
Re = 100.
viscosity = 1. / Re
grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
max_velocity = 5. # estimate (not prescribed)
dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, viscosity, grid) 

newton_solver = nt_sp.rpoSolverSpectral(dt_stable, grid, Re, nmax_hook=10, eps_newt=1e-10)


count = 0
for file_name in array_files:
  guesses = np.load(file_name).astype('float64') # tensorflow single prec output -> promote to double
  for guess in guesses:
    T = T_burst
    shift = shift_burst
    omega0 = jnp.fft.rfftn(guess.reshape((Nx, Ny)))

    po_guess_spectral = nt_sp.poGuessSpectral(omega0, T, shift)
  
    po_update = newton_solver.iterate(po_guess_spectral)
    if po_update.newt_resi_history[-1] < newton_solver.eps_newt:
      try:
        np.save('success_' + str(count) + '_spec_array.npy', po_update.omega_rft_out)
        np.save('success_' + str(count) + '_spec_meta.npy', np.array([po_update.T_out, 
                                                                    po_update.shift_out,
                                                                    po_update.newt_resi_history[-1]]))
        count += 1
      except:
        print("Erroneous convergence. Moving on.")

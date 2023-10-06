""" TODO -- move time advancement to specfic spectral routine. """
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

import pickle
import matplotlib.pyplot as plt

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral

from jax.config import config
config.update("jax_enable_x64", True)

import typing
from typing import Callable, Tuple, List, Union
Array = Union[np.ndarray, jnp.ndarray]

import interact_jaxcfd_dtypes as glue
import newton_spectral as nt_sp

po_check_file = 'sanity_po538_shift0.obj'

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

newton_solver = nt_sp.rpoSolverSpectral(dt_stable, grid, Re, nmax_hook=10, eps_newt=1e-10)

with open(po_check_file, 'rb') as f:
  guesses = pickle.load(f)

new_convs = {}
count = 0
for g_num in guesses:
  #omega0 = guesses[g_num].omega_init
  #T = guesses[g_num].T_init
  #shift = guesses[g_num].shift_init
  v0 = guesses[g_num].u_out
  T = guesses[g_num].T_out
  shift = guesses[g_num].shift_out

  umean, vmean = glue.mean_flows(v0)
  shift_update = shift - umean * T

  vorticity0 = cfd.finite_differences.curl_2d(v0).data
  omega0 = jnp.fft.rfftn(vorticity0)

  po_guess_spectral = nt_sp.poGuessSpectral(omega0, T, shift_update)
  
  po_update = newton_solver.iterate(po_guess_spectral)
  if po_update.newt_resi_history[-1] < newton_solver.eps_newt:
    new_convs[count] = po_update
    count += 1
    with open('newt_spec_convs.obj', 'wb') as f:
      pickle.dump(new_convs, f)

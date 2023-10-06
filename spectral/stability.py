""" Arnoldi for stability of periodic solutions """ 
import typing
from typing import Callable, Tuple, List, Union

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
Array = jnp.ndarray

from functools import partial

import jax_cfd.base  as cfd
import interact_spectral as insp 
import time_forward_map_spectral as tfm
import newton_spectral as nt_sp

from scipy.sparse.linalg import eigs as arp_eigs
from scipy.sparse.linalg import LinearOperator

def _timestep_monodromy(
    X: Array, # physical space shape (Ndof,) 
    rpo: nt_sp.poGuessSpectral,
    grid: cfd.grids.Grid,
    forward_map: Callable[[Array], Array], 
    eps_fd: float=1e-7
) -> Array:
  omega_rpo_rft = rpo.omega_rft_out
  shift = rpo.shift_out
  
  Nx, Ny = grid.shape
  X_rft = jnp.fft.rfftn(X.reshape((Nx, Ny)))

  # TODO replace finite difference jacobian estimate with jax
  eps_new = eps_fd * norm(omega_rpo_rft.reshape((-1,))) / norm(X_rft.reshape((-1,)))

  omega_X_T = forward_map(omega_rpo_rft + eps_new * X_rft.reshape(omega_rpo_rft.shape))

  AX = (1./eps_new) * (insp.x_shift_fourier_coeffs(omega_X_T, grid, shift) - omega_rpo_rft)
  AX_phys = jnp.fft.irfftn(AX).reshape((-1,))
  return AX_phys

def compute_rpo_stability(
    rpo: nt_sp.poGuessSpectral,
    grid: cfd.grids.Grid,
    dt_stable: float,
    Re: float,
    N_eig: int=50,
    eps_fd: float=1e-7
) -> Tuple[Array]:
  dt_exact = rpo.T_out / int(rpo.T_out / dt_stable)
  Nt = int(rpo.T_out / dt_exact)
  forward_map = tfm.generate_time_forward_map(dt_exact, Nt, grid, 1./Re)
  Ndof = grid.shape[0] * grid.shape[1]
  
  timestepper = partial(_timestep_monodromy, rpo=rpo, grid=grid, forward_map=forward_map, eps_fd=eps_fd)
  AX = LinearOperator(shape=(Ndof, Ndof), matvec=timestepper, dtype='float64')

  vel_ar = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(42), grid, 1., 4)
  vort_pert = cfd.finite_differences.curl_2d(vel_ar).data
  vort_pert_rft = jnp.fft.rfftn(vort_pert)
  
  starting_fourier = rpo.omega_rft_out + 0.01 * vort_pert_rft
  starting_vec = jnp.fft.irfftn(starting_fourier).reshape((-1,))
  
  floquet_mults, floquet_eigenv = arp_eigs(AX, k=N_eig, v0=starting_vec)
  return floquet_mults, floquet_eigenv 

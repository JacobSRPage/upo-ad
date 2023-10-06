import typing
from typing import Callable, Tuple, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

Array = Union[np.ndarray, jnp.ndarray]

import jax_cfd.base  as cfd
import arnoldi as ar

def x_shift(
    field: Array,
    grid: cfd.grids.Grid,
    x_shift: float
) -> Array:
  """ Shift field along a continuous symmetry 
      Assumes data is periodic, uniform grid etc 
      At the moment can shift only along the first axis (x) 
      [Continuous shifts in other directions will need a rewrite] """
  # TODO general method for FT of 3D arrays * phase shift
  field_rft = jnp.fft.rfftn(field)
  axis = 0 # shift along x only at present
  k = 2 * jnp.pi * jnp.fft.fftfreq(grid.shape[axis], grid.step[axis])
  field_phase_shift = (field_rft.T * jnp.exp(1j * k * x_shift)).T
  return jnp.fft.irfftn(field_phase_shift)

def x_derivative(
    field: Array,
    grid: cfd.grids.Grid
) -> Array:
  field_rft = jnp.fft.rfftn(field)
  axis = 0 # shift along x only at present
  k = 2 * jnp.pi * jnp.fft.fftfreq(grid.shape[axis], grid.step[axis])
  dfield_fft = (1j * field_rft.T * k).T
  return jnp.fft.irfftn(dfield_fft)

def x_shift_fourier_coeffs(
    field: Array,
    grid: cfd.grids.Grid,
    x_shift: float
) -> Array:
  """ Shift field along a continuous symmetry 
      Assumes data is periodic, uniform grid etc 
      At the moment can shift only along the first axis (x) 
      [Continuous shifts in other directions will need a rewrite] """
  # TODO general method for FT of 3D arrays * phase shift
  axis = 0 # shift along x only at present
  k = 2 * jnp.pi * jnp.fft.fftfreq(grid.shape[axis], grid.step[axis])
  field_phase_shift = (field.T * jnp.exp(1j * k * x_shift)).T
  return field_phase_shift

def x_derivative_fourier_coeffs(
    field: Array,
    grid: cfd.grids.Grid
) -> Array:
  axis = 0 
  k = 2 * jnp.pi * jnp.fft.fftfreq(grid.shape[axis], grid.step[axis])
  dfield_fft = (1j * field.T * k).T
  return dfield_fft

def y_derivative_fourier_coeffs(
    field: Array,
    grid: cfd.grids.Grid
) -> Array:
  axis = 1 
  k = 2 * jnp.pi * jnp.fft.rfftfreq(grid.shape[axis], grid.step[axis])
  dfield_fft = 1j * field * k
  return dfield_fft

def rhs_equations(
    vort_rft: jnp.ndarray,
    grid: cfd.grids.Grid,
    Re: float,
    n_kol_waves: int=4
) -> Array:
  """ NEEDS TESTING -- DOESN'T WORK YET. """
  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(grid.shape[0], grid.step[0])
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(grid.shape[1], grid.step[1])
  
  kx_mesh, ky_mesh = np.meshgrid(all_kx, all_ky)
  kx_mesh = kx_mesh.T
  ky_mesh = ky_mesh.T

  diffusion_rft = -(1. / Re) * (kx_mesh ** 2 + ky_mesh ** 2) * vort_rft 
  diffusion = jnp.fft.irfftn(diffusion_rft)  

  _, u_rft, v_rft = compute_streamfunction_and_vel_fourier(vort_rft, grid)
  u = jnp.fft.irfftn(u_rft)
  v = jnp.fft.irfftn(v_rft)
  
  domegadx_rft = x_derivative_fourier_coeffs(vort_rft, grid)
  domegady_rft = y_derivative_fourier_coeffs(vort_rft, grid)
  domegadx = jnp.fft.irfftn(domegadx_rft)
  domegady = jnp.fft.irfftn(domegady_rft)

  advection = u * domegadx + v * domegady

  forcing_term_fn = lambda x, y: -n_kol_waves * jnp.cos(n_kol_waves * y)
  y_grid = [0 + j * grid.step[1] for j in range(grid.shape[1])] 

  forcings = [forcing_term_fn(0, y) for y in y_grid]
  forcing_ar = jnp.asarray(forcings)

  return -advection + diffusion + forcing_ar

def compute_streamfunction_and_vel_fourier(
    vort_rft: jnp.ndarray,
    grid: cfd.grids.Grid
) -> Tuple[Array]:
  all_kx = 2 * jnp.pi * jnp.fft.fftfreq(grid.shape[0], grid.step[0])
  all_ky = 2 * jnp.pi * jnp.fft.rfftfreq(grid.shape[1], grid.step[1])
  
  kx_mesh, ky_mesh = np.meshgrid(all_kx, all_ky)
  kx_mesh = kx_mesh.T
  ky_mesh = ky_mesh.T
  
  np.seterr(divide='ignore') # will happen for 0,0 wavenumber
  psik = vort_rft / (kx_mesh ** 2 + ky_mesh ** 2)
  psik = psik.at[0,0].set(0.)
  
  uk =  1j * ky_mesh * psik
  vk = -1j * kx_mesh * psik
  return psik, uk, vk

def dissipation_spectral(
  vort_rft: Array, 
  Re: float,
  n_kol_waves: int = 4
) -> float:
  """ Assume vort_rft is not normalised
      Kolmogorov flow on 2 pi x 2 pi box. """
  Nx, Ny_2_1 = vort_rft.shape 
  Ny = 2 * (Ny_2_1 - 1)
  
  vort_copy = vort_rft[:, 1:]  # missing entries from rft needed in sum
  diss = jnp.sum(jnp.abs(vort_rft / (Nx * Ny)) ** 2) + \
    jnp.sum(jnp.abs(vort_copy / (Nx * Ny)) ** 2) 
  diss /= Re
  
  diss_laminar = Re / (2 * n_kol_waves ** 2)
  return diss / diss_laminar

def energy_spectral(
  vort_rft: jnp.ndarray,
  Re: float,
  grid: cfd.grids.Grid,
  n_kol_waves: int=4
) -> float:
  """ Assume vort_rft is not normalised
      Kolmogorov flow on 2 pi x 2 pi box. """
  Nx, Ny_2_1 = vort_rft.shape
  Ny = 2 * (Ny_2_1 - 1)

  _, uk, vk = compute_streamfunction_and_vel_fourier(vort_rft, grid)
  uk_copy = uk[:, 1:]
  vk_copy = vk[:, 1:] # missing entries from rft needed in sum
  
  energy = jnp.sum(jnp.abs(uk / (Nx * Ny)) ** 2) + jnp.sum(jnp.abs(uk_copy / (Nx * Ny)) ** 2) + \
    jnp.sum(jnp.abs(vk / (Nx * Ny)) ** 2) + jnp.sum(jnp.abs(vk_copy / (Nx * Ny)) ** 2)
  energy /= 2.
  
  energy_laminar = Re ** 2 / (4 * n_kol_waves ** 4)
  return energy / energy_laminar

def production_spectral(
  vort_rft: jnp.ndarray,
  Re: float,
  grid: cfd.grids.Grid,
  n_kol_waves: int=4
) -> float:
  """ Assume vort_rft is not normalised
      Kolmogorov flow on 2 pi x 2 pi box. """
  Nx, Ny_2_1 = vort_rft.shape
  Ny = 2 * (Ny_2_1 - 1)
  
  _, uk, _ = compute_streamfunction_and_vel_fourier(vort_rft, grid)
  uk_n = uk[0, n_kol_waves]
  
  prod = -jnp.imag(uk_n) / (Nx * Ny)
  diss_laminar = Re / (2 * n_kol_waves ** 2)
  return prod / diss_laminar

def mask_fourier_coeffs(
    field: Array,
    Nmax: int
) -> Array:
  return NotImplementedError

def state_vector_extra_parameters(
    u: Array, 
    extra_variables: List[float]
) -> Array:
  u_vec = u.reshape((-1,)) 
  return jnp.append(u_vec, jnp.asarray(extra_variables))

def grad_with_extras_vec(
    u: Array, 
    extra_variables: List[float],
    loss_fn: Callable[[Tuple[Array], float], float]
) -> Array:
  grads = jax.grad(loss_fn, argnums=[j for j in range(len(extra_variables)+1)])(u, *extra_variables)
  grad_u = grads[0]
  other_grads = grads[1:]
  return state_vector_extra_parameters(grad_u, [*other_grads])

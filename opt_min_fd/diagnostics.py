""" Derived quantities """ 
import numpy as np
import jax.numpy as jnp
from jax.numpy.fft import rfftn

import jax_cfd.base  as cfd

from typing import Tuple, List, Union
Array = Union[np.ndarray, jnp.ndarray]

def vol_integral_2D(
  field: Array,
  grid: cfd.grids.Grid
) -> float:
  """ Assumes 2D array """
  field_int_x = jnp.trapz(field, dx=grid.step[0], axis=0)
  return jnp.trapz(field_int_x, dx=grid.step[1], axis=0)

def compute_vort(
    vel_field: Tuple[cfd.grids.GridVariable]
) -> Array:
  """ Assume a uniform grid """
  vort_gv = cfd.finite_differences.curl_2d(vel_field) 
  return vort_gv.data 

def compute_diss_snapshot(
    vel_field: Tuple[cfd.grids.GridVariable],
    Re: float,
    n_kol_waves: int=4
) -> float:
  """ Kolmogorov flow on 2pi x 2pi. """
  vort_field = cfd.finite_differences.curl_2d(vel_field).data
  Nx, Ny = vort_field.shape

  vort_rft = rfftn(vort_field) 
  vort_copy = vort_rft[:, 1:]
  diss = jnp.sum(jnp.abs(vort_rft / (Nx * Ny)) ** 2) / (4 * jnp.pi ** 2) + \
    jnp.sum(jnp.abs(vort_copy / (Nx * Ny)) ** 2) / (4 * jnp.pi ** 2)
  
  diss_laminar = Re / (2 * n_kol_waves ** 2)
  return diss / diss_laminar

def compute_energy_snapshot(
    vel_field: Tuple[cfd.grids.GridVariable],
    grid: cfd.grids.Grid,
    Re: float,
    n_kol_waves: int=4
) -> float:
  """ Kolmogorov flow on 2pi x 2pi. """
  energy_density = 0.5 * (
    vel_field[0].data ** 2 + 
    vel_field[1].data ** 2
  )

  total_energy = vol_integral_2D(energy_density, grid)
  vol_avg_energy = total_energy / (2 * jnp.pi) ** 2
  
  energy_laminar = Re ** 2 / (4 * n_kol_waves ** 4)
  return vol_avg_energy / energy_laminar
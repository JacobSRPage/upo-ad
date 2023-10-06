""" Routines to interact with jax-cfd [more] """
import typing
from typing import Callable, Tuple, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

Array = Union[np.ndarray, jnp.ndarray]
import jax.numpy.fft as ft

import jax_cfd.base  as cfd

def files_to_gv_tuple(
    u_ar_file: str,
    grid: cfd.grids.Grid
) -> Tuple[cfd.grids.GridVariable]:
  u_ar = np.load(u_ar_file)
  offsets = [(1., 0.5), (0.5, 1.)]
  bc = cfd.boundaries.HomogeneousBoundaryConditions(types=(('periodic', 'periodic'), ('periodic', 'periodic')))
  return jnp_to_gv_tuple(u_ar, offsets, grid, bc)

def jnp_to_gv_tuple(
    u_jnp: Array, 
    offsets: List[Tuple], 
    grid: cfd.grids.Grid, 
    bc: cfd.grids.BoundaryConditions
) -> Tuple[cfd.grids.GridVariable]:
  gv_list = []
  for i in range(u_jnp.shape[-1]):
    garr = cfd.grids.GridArray(u_jnp[..., i], offsets[i], grid)
    gv_list.append(cfd.grids.GridVariable(garr, bc))
  return tuple(gv_list)


def gv_tuple_to_jnp(
    gv_tuple: Tuple[cfd.grids.GridVariable]
) -> Tuple[Array, List[Tuple]]:
  u_jnp = jnp.stack([gv.data for gv in gv_tuple], axis=-1)
  offsets = [gv.offset for gv in gv_tuple]
  return u_jnp, offsets


def x_shift_field(
    field: Tuple[cfd.grids.GridVariable], 
    x_shift: float
) -> Tuple[cfd.grids.GridVariable]:
  """ Shift field along a continuous symmetry 
      Assumes data is periodic, uniform grid etc 
      At the moment can shift only along the first axis (x) 
      [Continuous shifts in other directions will need a rewrite] """
  # TODO general method for FT of 3D arrays * phase shift
  axis = 0 # shift along x only at present
  
  shifted_field = ()
  for v in field:
    k = 2 * jnp.pi * ft.fftfreq(v.grid.shape[axis], v.grid.step[axis])  
    vel_fft = ft.fft(v.data, axis=axis).T
    vel_fft_phase_shift = (vel_fft * jnp.exp(1j * k * x_shift)).T
    vel_shifted = jnp.real(ft.ifft(vel_fft_phase_shift, axis=axis))

    vel_shifted_array = cfd.grids.GridArray(vel_shifted, v.offset, v.grid)
    shifted_field += (cfd.grids.GridVariable(vel_shifted_array, v.bc),)        
  return shifted_field


# TODO Combine derivatives for general method.
def x_derivative(
    field: Tuple[cfd.grids.GridVariable]
) -> Tuple[cfd.grids.GridVariable]:
  axis = 0 # shift along x only at present
  
  d_field = ()
  for v in field:
    k = 2 * jnp.pi * ft.fftfreq(v.grid.shape[axis], v.grid.step[axis]) 
    vel_fft = ft.fft(v.data, axis=axis).T
    d_vel_fft = (1j * vel_fft * k).T
    d_vel = jnp.real(ft.ifft(d_vel_fft, axis=axis))
    d_vel_array = cfd.grids.GridArray(d_vel, v.offset, v.grid)
    d_field += (cfd.grids.GridVariable(d_vel_array, v.bc),)
  return d_field

def y_derivative(
    field: Tuple[cfd.grids.GridVariable]
) -> Tuple[cfd.grids.GridVariable]:
  axis = 1 
  
  d_field = ()
  for v in field:
    k = 2 * jnp.pi * ft.fftfreq(v.grid.shape[axis], v.grid.step[axis]) 
    vel_fft = ft.fft(v.data, axis=axis)
    d_vel_fft = 1j * vel_fft * k
    d_vel = jnp.real(ft.ifft(d_vel_fft, axis=axis))
    
    d_vel_array = cfd.grids.GridArray(d_vel, v.offset, v.grid)
    d_field += (cfd.grids.GridVariable(d_vel_array, v.bc),)
  return d_field


def mean_flows(
    field: Tuple[cfd.grids.GridVariable], 
) -> List[float]:
  """ Compute mean U and V """
  mean_vels = []
  for v in field:
    vel_fft = ft.fftn(v.data)
    Nx, Ny = vel_fft.shape
    mean_vels.append(jnp.real(vel_fft[0,0] / (Nx * Ny)))
  return mean_vels


def state_vector(
    u: Tuple[cfd.grids.GridVariable]
) -> Array:
  u_field = jnp.stack([gv.data for gv in u], axis=-1)
  return u_field.flatten() 


def state_vector_extra_parameters(
    u: Tuple[cfd.grids.GridVariable], 
    extra_variables: List[float]
) -> Array:
  u_vec = state_vector(u)
  return jnp.append(u_vec, jnp.asarray(extra_variables))


def error_l2(
    u: Tuple[cfd.grids.GridVariable],
    u_ref: Tuple[cfd.grids.GridVariable]
) -> float:
  return la.norm(state_vector(u) - 
                 state_vector(u_ref)) / la.norm(state_vector(u_ref))


def grad_u_with_extras_vec(
    u: Tuple[cfd.grids.GridVariable], 
    extra_variables: List[float],
    loss_fn: Callable[[Tuple[cfd.grids.GridVariable], float], float]
) -> Array:
  grads = jax.grad(loss_fn, argnums=[j for j in range(len(extra_variables)+1)])(u, *extra_variables)
  grad_u = grads[0]
  other_grads = grads[1:]
  return state_vector_extra_parameters(grad_u, [*other_grads])

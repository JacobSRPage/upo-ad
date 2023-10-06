""" Compare the guess and the converged periodic orbit """
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import pickle
import jax.numpy as jnp

import jax_cfd.base  as cfd
import time_forward_map as tfm
import interact_jaxcfd_dtypes as glue
import diagnostics as dg 

import loss_functions as lf

file_name = 'guess_target_T8_ada035_N600.obj'
soln_indx = 0
n_snap = 4
offset = 0 #.5 # did we shift the time origin in the search?

# need to match the below!! <- should fix in a poGuess really
Nx = 256
Ny = 256
Lx = 2 * jnp.pi
Ly = 2 * jnp.pi

n_waves = 4 # number of forcing wavelengths
viscosity = 1. / 40.
density = 1.


solns_file = open(file_name, 'rb')
guesses = pickle.load(solns_file)
print([key for key in guesses])
guess_to_view = guesses[soln_indx]
print("Guess period: ", guess_to_view.T_init)
print("Guess shift: ", guess_to_view.shift_init)
print("Guess starting loss: ", guess_to_view.guess_loss)



grid = cfd.grids.Grid((Nx, Ny), domain=((0, Lx), (0, Ly)))
max_velocity = 5. # estimate (not prescribed)
dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, viscosity, grid) / 2. # should take care here
kolmogorov = cfd.forcings.kolmogorov_forcing(grid, k=n_waves)

forward_map = partial(tfm.apply_forward_map_T, grid=grid, viscosity=viscosity, 
                      density=density, max_velocity=max_velocity, forcing=kolmogorov)

#### sanity checks on the loss -- is diffrax working as expected?
#loss_check = lf.loss_fn_T(guess_to_view.u_init, 
#                          guess_to_view.T_init,
#                          guess_to_view.shift_init,
#                          apply_forward_map = forward_map)
#
#def convect(v): 
#  return tuple(cfd.advection.advect_linear(u, v, dt_stable) for u in v)
#
#step_fn = cfd.equations.semi_implicit_navier_stokes(
#      density=density,
#      viscosity=viscosity,
#      dt=dt_stable,
#      grid=grid,
#      convect=convect,
#      forcing=kolmogorov
#    )
#advance_velocity_fn = tfm.advance_velocity_module(step_fn, dt_stable, max_steps=2 * int(guess_to_view.T_init / dt_stable))
#
#
#loss_diff = lf.loss_fn_diffrax(guess_to_view.u_init, 
#                          guess_to_view.T_init,
#                          guess_to_view.shift_init,
#                          forward_map = advance_velocity_fn)
#print("Loss original: ", loss_check)
#print("Loss diffrax: ", loss_diff)


T_int = guess_to_view.T_init
vis_t = [j * T_int / n_snap for j in range(n_snap + 1)] # want to vis the end
guess_snaps = [guess_to_view.u_init]
dt_j = [t_j - t_j_1 for t_j, t_j_1 in zip(vis_t[1:], vis_t[:-1])]
for dt in dt_j:
  guess_snaps.append(forward_map(guess_snaps[-1], dt))

fig = plt.figure(figsize=(4 * (n_snap+1), 4))
col = 1
for gs in guess_snaps:
  ax1 = fig.add_subplot(1, n_snap+1, col)
  ax1.contourf(dg.compute_vort(gs).T, 51)
  ax1.set_xticks([])
  ax1.set_yticks([])

  col += 1
fig.tight_layout()
plt.show()

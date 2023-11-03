from jax.config import config
config.update("jax_enable_x64", True)

import copy

import jax.numpy as jnp
import jax_cfd.base as cfd

import interact_spectral as insp
import newton_spectral as nt_sp

#  setup problem 
Nx = 256
Ny = 256
L = 2 * jnp.pi

# number of states to be converged
N_states = 50

# Reynolds number of solution to be loaded, and perturbing Re 
Re = 400.
delta_Re_init = 10. 
soln_file_path = 'success_l1_20_30_0_spec_array.npy'
meta_file_path = 'success_l1_20_30_0_spec_meta.npy'


output_file_front = 'euler_soln_'


upo_rft = jnp.load(soln_file_path)
upo_meta = jnp.load(meta_file_path)

# create original upo guess, and a copy to converged at target d_Re
upo_start = nt_sp.poGuessSpectral(upo_rft, upo_meta[0], upo_meta[1], n_shift_reflects=0)
upo_perturbed = copy.deepcopy(upo_start)
upo_start.record_outcome(upo_rft, upo_start.T_init, upo_start.shift_init, None, None)

# create newton/branch continuation solves
grid = cfd.grids.Grid((Nx, Ny), domain=((0, L), (0, L)))
max_velocity = 5. # estimate 
dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, 1. / Re, grid) 

newton_solver = nt_sp.rpoSolverSpectral(grid)
branch_continuation_solver = nt_sp.rpoBranchContinuation(grid)

# converge perturbed solution to start the contunuation process 
print("Converging starting perturbed solution")
Re_new = Re + delta_Re_init
upo_perturbed = newton_solver.iterate(upo_perturbed, Re_new, dt_stable)
nt_sp.write_out_soln_info(output_file_front, 1, upo_perturbed, Re_new)

# perform arclength continuation
states = [upo_start, upo_perturbed]
Re_store = [Re, Re_new]
for j in range(2, N_states):
  new_soln, new_Re = branch_continuation_solver.iterate(states[-1], 
                                                        states[-2], 
                                                        Re_store[-1], 
                                                        Re_store[-2], 
                                                        dt_stable)
  if new_soln.newt_resi_history[-1] > branch_continuation_solver.eps_newt:
    print("Failed to converge new solution. Terminating.")
    break
  nt_sp.write_out_soln_info(output_file_front, j, new_soln, new_Re)
  states.append(new_soln)
  Re_store.append(new_Re)
  dt_stable = cfd.equations.stable_time_step(max_velocity, 0.5, 1. / Re_store[-1], grid)
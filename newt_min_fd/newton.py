from typing import Tuple, List, Union
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from functools import partial

Array = Union[np.ndarray, jnp.ndarray]

import jax_cfd.base  as cfd
import arnoldi as ar
import interact_jaxcfd_dtypes as glue 
import time_forward_map as tfm

class poGuess:
  def __init__(
      self, 
      u: Tuple[cfd.grids.GridVariable], 
      T: float, 
      shift: float,
      n_shift_reflects: int=0,
      guess_loss: float=None
  ):
    # TODO convert for pure PO? Or shift = None?
    self.u_init = u
    self.T_init = T
    self.shift_init = shift

    self.n_shift_reflects = n_shift_reflects
    self.guess_loss = guess_loss

  def record_outcome(
      self,
      u_out: Tuple[cfd.grids.GridVariable],
      T_out: float,
      shift_out: float,
      newton_residual_history: List[float],
      newton_iterations: int
  ):
    self.u_out = u_out
    self.T_out = T_out
    self.shift_out = shift_out
    self.newt_resi_history = newton_residual_history
    self.newt_it = newton_iterations

class newtonBase:
  def __init__(
      self, 
      dt_stable: float, 
      grid: cfd.grids.Grid,
      Re: float,
      eps_newt: float=1e-10, 
      eps_gm: float=1e-3, 
      eps_fd: float=1e-7,
      nmax_newt: int=100, 
      nmax_hook: int=10,
      nmax_gm: int=100, 
      Delta_rel: float=0.1
  ):
    """ Delta_start * norm(u) will be the size of the Hookstep constraint. """
    self.current_tfm = None # will repeatedly build time forward map
    self.Re = Re
    self.viscosity = 1. / Re # legacy
    self.grid = grid

    self.eps_newt = eps_newt # Newton step convergence
    self.eps_gm = eps_gm # GMRES Krylov convergence
    self.eps_fd = eps_fd # step size for forward difference approx to A * q
    self.nmax_newt = nmax_newt # max newton iterations
    self.nmax_hook = nmax_hook
    self.nmax_gm = nmax_gm # max steps for GMRES

    self.dt_stable = dt_stable
    self.Delta_rel = Delta_rel

  def _initialise_guess(
      self, 
      po_guess: poGuess
  ):
    return NotImplementedError
  
  def _jit_tfm(
      self
  ):
    """ Create time forward map (minimal number of times) """
    dt_exact = self.T_current / int(self.T_current / self.dt_stable)
    Nt = int(self.T_current / dt_exact)
    self.current_tfm = tfm.generate_time_forward_map(dt_exact, Nt, self.grid, self.Re)

  def _timestep_DNS(
      self, 
      u0: Tuple[cfd.grids.GridVariable], 
      T_march: float
  ) -> Tuple[cfd.grids.GridVariable]:
    if T_march != self.T_current:
      self.T_current = T_march
      self._jit_tfm()

    uT = self.current_tfm(u0)
    return uT
  
  def iterate(
      self, 
      po_guess: poGuess
  ) -> poGuess:
    return NotImplementedError
      
  def _timestep_A(
      self, 
      eta_w_T: Array
  ) -> Array:
    return NotImplementedError
  
  def _jnp_norm(
      self, 
      gv: Tuple[cfd.grids.GridVariable]
  ):
    g_arr = glue.state_vector(gv)
    return la.norm(g_arr)

  def _update_F(
      self
  ):
    return NotImplementedError


class upoSolver(newtonBase):
  def _initialise_guess(
      self, 
      po_guess: poGuess
  ):
    self.u_guess = po_guess.u_init
    self.T_guess = po_guess.T_init

    self.T_current = 0. # keep track to re-jit tfm 

    self.Delta_start = self.Delta_rel * self._jnp_norm(self.u_guess) 

    u_ar, offsets = glue.gv_tuple_to_jnp(self.u_guess)
    self.original_shape = u_ar.shape
    self.Ndof = u_ar.size
    
    self.offsets = offsets
    self.bc = self.u_guess[0].bc

    self._update_F() # S_x u(u0,t) - u0
    self._update_du_dt() # du0/dt and d S_x uT/dT

 
  def iterate(
      self, 
      po_guess: poGuess
  ) -> poGuess:
    self._initialise_guess(po_guess)

    newt_res = la.norm(self.F)
    res_history = []
    newt_count = 0
    while la.norm(self.F) / self._jnp_norm(self.u_guess) > self.eps_newt: # <<< FIX with T, a 
      kr_basis, gm_res, _ = ar.gmres(self._timestep_A, 
                                     -self.F, 
                                     self.eps_gm, 
                                     self.nmax_gm)
      dx, _ = ar.hookstep(kr_basis, 2 * self.Delta_start)
      
      u_guess_update_arr = glue.state_vector(self.u_guess) + dx[:self.Ndof]
      self.u_guess = glue.jnp_to_gv_tuple(u_guess_update_arr.reshape(self.original_shape), 
                                          self.offsets, 
                                          self.grid, 
                                          self.bc)
      self.T_guess += dx[-1]

      self._update_F()
      self._update_du_dt()

      newt_new = la.norm(self.F)

      # (more) hooksteps if reqd
      Delta = self.Delta_start
      hook_count = 1
      print("old res: ", newt_res, "new_res: ", newt_new)

      if newt_new > newt_res:
        u_local = glue.state_vector(self.u_guess) - dx[:self.Ndof]
        T_local = self.T_guess - dx[-1]
        print("Starting Hookstep... ")
        while newt_new > newt_res:
          dx, _ = ar.hookstep(kr_basis, Delta)
          self.u_guess = glue.jnp_to_gv_tuple((u_local +  dx[:self.Ndof]).reshape(self.original_shape), 
                                              self.offsets, 
                                              self.grid, 
                                              self.bc)
          self.T_guess = T_local + dx[-1]

          self._update_F()
          self._update_du_dt()

          newt_new = la.norm(self.F)
          Delta /= 2.
          hook_count += 1
        print("# hooksteps:", hook_count)
      print("Current Newton residual: ", la.norm(self.F) / self._jnp_norm(self.u_guess))
      print("T guess: ", self.T_guess)
      newt_res = newt_new
      res_history.append(newt_res / self._jnp_norm(self.u_guess))
      newt_count += 1
      
      if newt_count > self.nmax_newt: 
        print("Newton count exceeded limit. Ending guess.")
        break
      if hook_count > self.nmax_hook:
        print("Hook steps exceeded limit. Ending guess.")
        break
      if self.T_guess < 0.:
        print("Negative period invalid. Ending guess.")
        break
    po_guess.record_outcome(self.u_guess, self.T_guess, None, res_history, newt_count)
    return po_guess
      
  def _timestep_A(
      self, 
      eta_w_T: Array
  ) -> Array:
    # TODO replace finite difference jacobian estimate with jacfwd
    eps_new = self.eps_fd * self._jnp_norm(self.u_guess) / la.norm(eta_w_T[:self.Ndof])

    # should clean up this back and forth
    array_to_timestep = glue.state_vector(self.u_guess) + eps_new * eta_w_T[:self.Ndof]
    gv_to_timestep = glue.jnp_to_gv_tuple(array_to_timestep.reshape(self.original_shape), 
                                          self.offsets, 
                                          self.grid, 
                                          self.bc)
    u_eta_T = self._timestep_DNS(gv_to_timestep, self.T_guess)

    u_diff_ar = glue.state_vector(u_eta_T) - glue.state_vector(self.u_T)
    Aeta = (1./eps_new) * u_diff_ar - eta_w_T[:self.Ndof]

    Aeta += glue.state_vector(self.duT_dT) * eta_w_T[-1]

    Aeta_w_T = np.append(Aeta, np.dot(glue.state_vector(self.du0_dt), eta_w_T[:self.Ndof]))
    return Aeta_w_T


  def _update_F(
      self
  ):
    self.u_T = self._timestep_DNS(self.u_guess, self.T_guess)
    u_T_arr = glue.state_vector(self.u_T)
    u_g_arr = glue.state_vector(self.u_guess)
    self.F = np.append(u_T_arr - u_g_arr, [0.]) # zero for T row

  def _update_du_dt(
      self
  ):
    self.du0_dt = self._compute_du_dt(self.u_guess) 
    self.duT_dT = self._compute_du_dt(self.u_T)
  

  def _compute_du_dt(
      self, 
      u_0: Tuple[cfd.grids.GridVariable]
  ) -> Tuple[cfd.grids.GridVariable]:
    dt_ = 10 * self.dt_stable
    u_out = self._timestep_DNS(u_0, dt_)
      
    dudt_arr = (glue.state_vector(u_out) - glue.state_vector(u_0)) / dt_
    dudt = glue.jnp_to_gv_tuple(dudt_arr.reshape(self.original_shape), 
                                self.offsets, 
                                self.grid, 
                                self.bc)
    return dudt


class rpoSolver(upoSolver):
  def _initialise_guess(
      self, 
      po_guess: poGuess
  ):
    self.u_guess = po_guess.u_init
    self.T_guess = po_guess.T_init
    if po_guess.shift_init == None:
      self.a_guess = 0.
    else:
      self.a_guess = po_guess.shift_init
    self.T_current = 0. # keep track for re-jitting tfm 

    self.Delta_start = self.Delta_rel * self._jnp_norm(self.u_guess) 

    u_ar, offsets = glue.gv_tuple_to_jnp(self.u_guess)
    self.original_shape = u_ar.shape
    self.Ndof = u_ar.size
    
    self.offsets = offsets 
    self.grid = self.u_guess[0].grid
    self.bc = self.u_guess[0].bc

    # initialise a shift reflect function
    self.shift_reflect_fn = partial(glue.shift_reflect_field, n_shift_reflects=po_guess.n_shift_reflects)

    self._update_F() # S_x T^n_sr u(u0,t) - u0
    self._update_du_dx() # du0/dx and d S_x T^n_sr uT/dx
    self._update_du_dt() # du0/dt and d S_x T^n_sr uT/dT

  
  def iterate(
      self, 
      po_guess: poGuess
  ) -> poGuess:
    self._initialise_guess(po_guess)

    newt_res = la.norm(self.F)
    res_history = []
    newt_count = 0
    hook_over = 0
    while la.norm(self.F) / self._jnp_norm(self.u_guess) > self.eps_newt: # <<< FIX with T, a 
      kr_basis, gm_res, _ = ar.gmres(self._timestep_A, -self.F, self.eps_gm, self.nmax_gm)
      dx, _ = ar.hookstep(kr_basis, 2*self.Delta_start)
      
      u_guess_update_arr = glue.state_vector(self.u_guess) + dx[:self.Ndof]
      self.u_guess = glue.jnp_to_gv_tuple(u_guess_update_arr.reshape(self.original_shape), 
                                          self.offsets, 
                                          self.grid, 
                                          self.bc)
      self.a_guess += dx[-2]
      self.T_guess += dx[-1]

      self._update_F()
      self._update_du_dx()
      self._update_du_dt()

      newt_new = la.norm(self.F)

      # (more) hooksteps if reqd
      Delta = self.Delta_start
      hook_count = 1
      hook_info = 0 
      print("old res: ", newt_res, "new_res: ", newt_new)
      if newt_new > newt_res:
        u_local = glue.state_vector(self.u_guess) - dx[:self.Ndof]
        a_local = self.a_guess - dx[-2]
        T_local = self.T_guess - dx[-1]
        print("Starting Hookstep... ")
        while newt_new > newt_res:
          dx, hook_info = ar.hookstep(kr_basis, Delta)
          self.u_guess = glue.jnp_to_gv_tuple((u_local +  dx[:self.Ndof]).reshape(self.original_shape), 
                                              self.offsets, 
                                              self.grid, 
                                              self.bc)
          self.a_guess = a_local + dx[-2]
          self.T_guess = T_local + dx[-1]

          self._update_F()
          self._update_du_dx()
          self._update_du_dt()

          newt_new = la.norm(self.F)
          Delta /= 2.
          hook_count += 1
          if hook_info == 'stop': 
            break # couldn't bound Delta
        print("# hooksteps:", hook_count)
      print("Current Newton residual: ", la.norm(self.F) / self._jnp_norm(self.u_guess))
      print("x-shift guess: ", self.a_guess)
      print("T guess: ", self.T_guess)
      newt_res = newt_new
      res_history.append(newt_res / self._jnp_norm(self.u_guess))
      newt_count += 1
      
      if newt_count > self.nmax_newt: 
        print("Newton count exceeded limit. Ending guess.")
        break
      if hook_info == 'stop':
        print("Could not obtain any valid trust region. Ending guess.")
        break
      if hook_count > self.nmax_hook:
        hook_over += 1
        if hook_over > 2:
          print("Hook steps exceeded limit. Ending guess.")
          break
        else:
          print("Hook went over. Allow ", hook_over, 'of', 2)
      if self.T_guess < 0.:
        print("Negative period invalid. Ending guess.")
        break
    po_guess.record_outcome(self.u_guess, self.T_guess, self.a_guess, res_history, newt_count)
    return po_guess
      
  def _timestep_A(
      self, 
      eta_w_T: Array
  ) -> Array:
    # TODO replace finite difference jacobian estimate with jacfwd
    eps_new = self.eps_fd * self._jnp_norm(self.u_guess) / la.norm(eta_w_T[:self.Ndof])

    # should clean up this back and forth
    array_to_timestep = glue.state_vector(self.u_guess) + eps_new * eta_w_T[:self.Ndof]
    gv_to_timestep = glue.jnp_to_gv_tuple(array_to_timestep.reshape(self.original_shape), 
                                          self.offsets, 
                                          self.grid, 
                                          self.bc)
    u_eta_T = self._timestep_DNS(gv_to_timestep, self.T_guess)

    array_to_shift = glue.state_vector(u_eta_T) - glue.state_vector(self.u_T)
    gv_to_shift = glue.jnp_to_gv_tuple(array_to_shift.reshape(self.original_shape), 
                                       self.offsets, self.grid, self.bc)
    Aeta = (1./eps_new) * glue.state_vector(self.shift_reflect_fn(glue.x_shift_field(gv_to_shift, self.a_guess))) - eta_w_T[:self.Ndof]

    Aeta += glue.state_vector(self.dSuT_dx) * eta_w_T[-2]
    Aeta += glue.state_vector(self.dSuT_dT) * eta_w_T[-1]

    Aeta_w_x = np.append(Aeta, np.dot(glue.state_vector(self.du0_dx), eta_w_T[:self.Ndof]))
    Aeta_w_x_T = np.append(Aeta_w_x, np.dot(glue.state_vector(self.du0_dt), eta_w_T[:self.Ndof]))
    return Aeta_w_x_T
 
  def _update_F(
      self
  ):
    self.u_T = self._timestep_DNS(self.u_guess, self.T_guess)
    shifted_u_T_arr = glue.state_vector(self.shift_reflect_fn(glue.x_shift_field(self.u_T, self.a_guess)))
    u_g_arr = glue.state_vector(self.u_guess)
    self.F = np.append(shifted_u_T_arr - u_g_arr, [0., 0.]) # zero for shift, T rows

  def _update_du_dt(
      self
  ):
    self.du0_dt = self._compute_du_dt(self.u_guess) 
    self.dSuT_dT = self._compute_du_dt(self.shift_reflect_fn(glue.x_shift_field(self.u_T, self.a_guess)))

  def _update_du_dx(
      self
  ):
    self.du0_dx = self._compute_du_dx(self.u_guess)
    self.dSuT_dx = self._compute_du_dx(self.shift_reflect_fn(glue.x_shift_field(self.u_T, self.a_guess)))

  def _compute_du_dx(
      self, 
      u_0: Tuple[cfd.grids.GridVariable]
  ) -> Tuple[cfd.grids.GridVariable]:
    return glue.x_derivative(u_0)
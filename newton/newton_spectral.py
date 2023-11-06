""" Have all routines have real input/output, ffts all inside;
    forward maps advance SPECTRAL coeffs  """
from typing import List, Union
from functools import partial

import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

Array = Union[np.ndarray, jnp.ndarray]

import jax_cfd.base  as cfd

from opt_newt_jaxcfd.newton import arnoldi as ar
from opt_newt_jaxcfd.interact_jaxcfd import interact_spectral as insp
from opt_newt_jaxcfd.interact_jaxcfd import time_forward_map_spectral as tfm

class poGuessSpectral:
  def __init__(
      self, 
      omega_rft: Array, 
      T: float, 
      shift: float,
      n_shift_reflects: int=0,
      guess_loss: float=None,
      trav_wave: bool=False
  ):
    """ omega_rft: spectral coefficients from rfftn(vorticity_physical) """
    self.omega_rft_init = omega_rft 
    self.T_init = T
    self.shift_init = shift

    self.n_shift_reflects = n_shift_reflects
    self.guess_loss = guess_loss

    if trav_wave == True:
      self.wavespeed_init = self.shift_init / self.T_init

  def record_outcome(
      self,
      omega_rft_out: Array,
      T_out: float,
      shift_out: float,
      newton_residual_history: List[float],
      newton_iterations: int,
      trav_wave: bool=False
  ):
    self.omega_rft_out = omega_rft_out
    self.T_out = T_out
    self.shift_out = shift_out
    self.newt_resi_history = newton_residual_history
    self.newt_it = newton_iterations

    if trav_wave == True:
      self.wavespeed_out = self.shift_out / self.T_out

class newtonBaseSpectral:
  def __init__(
      self, 
      grid: cfd.grids.Grid,
      eps_newt: float=1e-10, 
      eps_gm: float=1e-3, 
      eps_fd: float=1e-7,
      nmax_newt: int=50, 
      nmax_hook: int=10,
      nmax_gm: int=100, 
      Delta_rel: float=0.1
  ):
    """ Delta_start * norm(u) will be the size of the Hookstep constraint. """
    self.current_tfm = None # will repeatedly rebuild time forward map 
    self.grid = grid

    self.eps_newt = eps_newt # Newton step convergence
    self.eps_gm = eps_gm # GMRES Krylov convergence
    self.eps_fd = eps_fd # step size for forward difference approx to A * q
    self.nmax_newt = nmax_newt # max newton iterations
    self.nmax_hook = nmax_hook
    self.nmax_gm = nmax_gm # max steps for GMRES

    self.Delta_rel = Delta_rel

  def _initialise_guess(
      self, 
      po_guess: poGuessSpectral
  ):
    return NotImplementedError

  def _jit_tfm(
      self
  ):
    """ Create time forward map minimal number of times """
    dt_exact = self.T_current / int(self.T_current / self.dt_stable)
    Nt = int(self.T_current / dt_exact)
    self.current_tfm = tfm.generate_time_forward_map(dt_exact, Nt, self.grid, self.viscosity)

  def _timestep_DNS(
      self, 
      omega_0: Array, # in physical space 
      T_march: float
  ) -> Array:
    if T_march != self.T_current:
      self.T_current = T_march
      self._jit_tfm() 
  
    omega_rft_0 = jnp.fft.rfftn(omega_0) 
    omega_rft_T = self.current_tfm(omega_rft_0)
    omega_T = jnp.fft.irfftn(omega_rft_T)
    return omega_T 
  
  def iterate(
      self, 
      po_guess: poGuessSpectral
  ) -> poGuessSpectral:
    return NotImplementedError
      
  def _timestep_A(
      self, 
      eta_w_T: Array
  ) -> Array:
    return NotImplementedError

  def norm_field(
      self,
      field: Array
  ) -> float:
    return la.norm(field.reshape((-1,)))

  def _update_F(
      self
  ):
    return NotImplementedError


class rpoSolverSpectral(newtonBaseSpectral):
  def _initialise_guess(
      self, 
      po_guess: poGuessSpectral,
      Re: float,
      dt_stable: float
  ):
    self.omega_guess = jnp.fft.irfftn(po_guess.omega_rft_init)
    self.T_guess = po_guess.T_init
    if po_guess.shift_init == None:
      self.a_guess = 0.
    else:
      self.a_guess = po_guess.shift_init
    self.original_shape = self.omega_guess.shape
    self.Re = Re
    self.viscosity = 1. / Re
    self.dt_stable = dt_stable

    self.T_current = 0. # keep track for re-jitting time forward map

    self.Delta_start = self.Delta_rel * self.norm_field(self.omega_guess) 

    self.Ndof = self.omega_guess.size

    # create shift reflect function 
    self.shift_reflect_fn = partial(insp.y_shift_reflect, 
                                    grid=self.grid, 
                                    n_shift_reflects=po_guess.n_shift_reflects)

    self._update_F() # S_x T^n_SR u(u0,t) - u0
    self._update_du_dx() # du0/dx and d S_x T^n_SR uT/dx
    self._update_du_dt() # du0/dt and d S_x T^n_SR uT/dT

  def iterate(
      self, 
      po_guess: poGuessSpectral,
      Re: float,
      dt_stable: float
  ) -> poGuessSpectral:
    self._initialise_guess(po_guess, Re, dt_stable)

    newt_res = la.norm(self.F)
    res_history = []
    newt_count = 0
    while la.norm(self.F) / self.norm_field(self.omega_guess) > self.eps_newt: # <<< FIX with T, a 
      kr_basis, _, _ = ar.gmres(self._timestep_A, -self.F, self.eps_gm, self.nmax_gm)
      dx, _ = ar.hookstep(kr_basis, 2 * self.Delta_start)
      
      omega_guess_update_arr = self.omega_guess.reshape((-1,)) + dx[:self.Ndof]
      self.omega_guess = omega_guess_update_arr.reshape(self.original_shape) 
      self.a_guess += dx[-2]
      self.T_guess += dx[-1]

      self._update_F()
      self._update_du_dx()
      self._update_du_dt()

      newt_new = la.norm(self.F)

      # (more) hooksteps if reqd
      Delta = self.Delta_start
      hook_count = 1
      print("old res: ", newt_res, "new_res: ", newt_new)
      if newt_new > newt_res:
        omega_local = self.omega_guess.reshape((-1,)) - dx[:self.Ndof]
        a_local = self.a_guess - dx[-2]
        T_local = self.T_guess - dx[-1]
        print("Starting Hookstep... ")
        while newt_new > newt_res:
          dx, hook_info = ar.hookstep(kr_basis, Delta)
          self.omega_guess = (omega_local + dx[:self.Ndof]).reshape(self.original_shape) 
          self.a_guess = a_local + dx[-2]
          self.T_guess = T_local + dx[-1]

          self._update_F()
          self._update_du_dx()
          self._update_du_dt()

          newt_new = la.norm(self.F)
          Delta /= 2.
          hook_count += 1
          if hook_info == 'stop': break # couldn't bound Delta
        print("# hooksteps:", hook_count)
      print("Current Newton residual: ", la.norm(self.F) / 
           self.norm_field(self.omega_guess))
      print("x-shift guess: ", self.a_guess)
      print("T guess: ", self.T_guess)
      newt_res = newt_new
      res_history.append(newt_res / self.norm_field(self.omega_guess))
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
    po_guess.record_outcome(jnp.fft.rfftn(self.omega_guess), self.T_guess, self.a_guess, res_history, newt_count)
    return po_guess
      
  def _timestep_A(
      self, 
      eta_w_T: Array
  ) -> Array:
    # TODO replace finite difference jacobian estimate with jacfwd
    eps_new = self.eps_fd * self.norm_field(self.omega_guess) / la.norm(eta_w_T[:self.Ndof])

    # should clean up this back and forth
    array_to_timestep = self.omega_guess.reshape((-1,)) + eps_new * eta_w_T[:self.Ndof]
    omega_eta_T = self._timestep_DNS(array_to_timestep.reshape(self.original_shape), self.T_guess)

    omega_to_shift = omega_eta_T - self.omega_T
    Aeta = (
      (1./eps_new) * self.shift_reflect_fn(insp.x_shift(omega_to_shift, self.grid, self.a_guess)).reshape((-1,)) - 
      eta_w_T[:self.Ndof]
      )

    Aeta += self.dSomegaT_dx.reshape((-1,)) * eta_w_T[-2]
    Aeta += self.dSomegaT_dT.reshape((-1,)) * eta_w_T[-1]

    Aeta_w_x = np.append(Aeta, np.dot(self.domega0_dx.reshape((-1,)), eta_w_T[:self.Ndof])) 
    Aeta_w_x_T = np.append(Aeta_w_x, np.dot(self.domega0_dt.reshape((-1,)).conj(), eta_w_T[:self.Ndof])) 
    return Aeta_w_x_T
 
  def _update_F(
      self
  ):
    self.omega_T = self._timestep_DNS(self.omega_guess, self.T_guess)
    shifted_omega_T_arr = self.shift_reflect_fn(insp.x_shift(self.omega_T, self.grid, self.a_guess)).reshape((-1,))
    omega_g_arr = self.omega_guess.reshape((-1,))
    self.F = np.append(shifted_omega_T_arr - omega_g_arr, [0., 0.]) # zero for shift, T rows

  def _update_du_dt(
      self
  ):
    self.domega0_dt = self._compute_domega_dt(self.omega_guess) 
    self.dSomegaT_dT = self._compute_domega_dt(self.shift_reflect_fn(insp.x_shift(self.omega_T, self.grid, self.a_guess)))

  def _update_du_dx(
      self
  ):
    self.domega0_dx = self._compute_domega_dx(self.omega_guess)
    self.dSomegaT_dx = self._compute_domega_dx(self.shift_reflect_fn(insp.x_shift(self.omega_T, self.grid, self.a_guess)))

  def _compute_domega_dx(
      self, 
      omega_0: Array 
  ) -> Array:
    return insp.x_derivative(omega_0, self.grid)
  

  def _compute_domega_dt(
      self, 
      omega_0: Array 
  ) -> Array:
    omega_rft = jnp.fft.rfftn(omega_0)
    return insp.rhs_equations(omega_rft, self.grid, self.Re)

class rpoBranchContinuation(rpoSolverSpectral):
  def _jit_tfm(
      self
  ):
    """ Create time forward map minimal number of times.
        Needs re-defining here due to viscosity changes. """
    dt_exact = self.T_current / int(self.T_current / self.dt_stable)
    Nt = int(self.T_current / dt_exact)
    self.current_tfm = tfm.generate_time_forward_map(dt_exact, Nt, self.grid, self.viscosity)

  def _timestep_DNS(
      self, 
      omega_0: Array, # in physical space 
      T_march: float, 
      viscosity_sim: float
  ) -> Array:
    if (T_march != self.T_current) or (viscosity_sim != self.viscosity):
      self.T_current = T_march
      self.viscosity = viscosity_sim
      self._jit_tfm() 
  
    omega_rft_0 = jnp.fft.rfftn(omega_0) 
    omega_rft_T = self.current_tfm(omega_rft_0)
    omega_T = jnp.fft.irfftn(omega_rft_T)
    return omega_T 

  def _initialise_guess(
      self, 
      po_conv_0: poGuessSpectral,
      po_conv__1: poGuessSpectral,
      Re_0: float,
      Re__1: float,
      dt_stable: float
      ):
    self.po_0 = po_conv_0
    self.po__1 = po_conv__1
    self.Re_0 = Re_0
    self.Re__1 = Re__1

    self.X_0 = np.append(
      jnp.fft.irfftn(self.po_0.omega_rft_out).reshape((-1,)),
      [self.po_0.shift_out, self.po_0.T_out, self.Re_0]
    )
    self.X__1 = np.append(
      jnp.fft.irfftn(self.po__1.omega_rft_out).reshape((-1,)),
      [self.po__1.shift_out, self.po__1.T_out, self.Re__1]
    )

    self.omega_guess = (jnp.fft.irfftn(self.po_0.omega_rft_out) + 
                        (jnp.fft.irfftn(self.po_0.omega_rft_out) - jnp.fft.irfftn(self.po__1.omega_rft_out)))
    self.T_guess = self.po_0.T_out + (self.po_0.T_out - self.po__1.T_out)
    self.a_guess = self.po_0.shift_out  + (self.po_0.shift_out - self.po__1.shift_out)
    self.Re_guess = self.Re_0 + (self.Re_0 - self.Re__1)

    self.dt_stable = dt_stable

    self._compute_dr()

    self.original_shape = self.omega_guess.shape

    # viscosity will change throughout requiring re-compiliation of forward map
    self.viscosity = 1. / self.Re_guess 
    self.T_current = 0. # keep track for re-jitting time forward map

    self.Delta_start = self.Delta_rel * self.norm_field(self.omega_guess) 

    self.Ndof = self.omega_guess.size

    if po_conv_0.n_shift_reflects != po_conv__1.n_shift_reflects:
      raise ValueError("Discrete shift reflects for the solutions must match!")
    self.n_shift_reflects = po_conv__1.n_shift_reflects
    # create shift reflect function 
    self.shift_reflect_fn = partial(insp.y_shift_reflect, 
                                    grid=self.grid, 
                                    n_shift_reflects=self.n_shift_reflects)
    
    self._update_dX_dr() # d X / dr (arclength derivative)
    self._update_F() # S_x T^n_SR u(u0,t) - u0
    self._update_du_dx() # du0/dx and d S_x T^n_SR uT/dx
    self._update_du_dt() # du0/dt and d S_x T^n_SR uT/dT
    self._update_du_dRe() # d S_x T^n_SR uT / dRe 

  def iterate(
      self, 
      po_conv_0: poGuessSpectral,
      po_conv__1: poGuessSpectral,
      Re_0: float,
      Re__1: float,
      dt_stable: float
      ) -> poGuessSpectral:
    self._initialise_guess(po_conv_0,
                           po_conv__1,
                           Re_0,
                           Re__1,
                           dt_stable)

    po_guess = poGuessSpectral(
      self.omega_guess,
      self.T_guess,
      self.a_guess,
      n_shift_reflects=self.n_shift_reflects
    )

    newt_res = la.norm(self.F)
    res_history = []
    newt_count = 0
    while la.norm(self.F) / self.norm_field(self.omega_guess) > self.eps_newt: # <<< FIX with T, a 
      kr_basis, _, _ = ar.gmres(self._timestep_A, -self.F, self.eps_gm, self.nmax_gm)
      dx, _ = ar.hookstep(kr_basis, 2 * self.Delta_start)
      
      omega_guess_update_arr = self.omega_guess.reshape((-1,)) + dx[:self.Ndof]
      self.omega_guess = omega_guess_update_arr.reshape(self.original_shape) 
      self.a_guess += dx[-3]
      self.T_guess += dx[-2]
      self.Re_guess += dx[-1]

      # update viscosity so that time forward map defined correctly
      self.viscosity = 1. / self.Re_guess

      self._update_dX_dr()
      self._update_F()
      self._update_du_dx()
      self._update_du_dt()
      self._update_du_dRe()

      newt_new = la.norm(self.F)

      # (more) hooksteps if reqd
      Delta = self.Delta_start
      hook_count = 1
      print("old res: ", newt_res, "new_res: ", newt_new)
      if newt_new > newt_res:
        omega_local = self.omega_guess.reshape((-1,)) - dx[:self.Ndof]
        a_local = self.a_guess - dx[-3]
        T_local = self.T_guess - dx[-2]
        Re_local = self.Re_guess - dx[-1]
        print("Starting Hookstep... ")
        while newt_new > newt_res:
          dx, hook_info = ar.hookstep(kr_basis, Delta)
          self.omega_guess = (omega_local + dx[:self.Ndof]).reshape(self.original_shape) 
          self.a_guess = a_local + dx[-3]
          self.T_guess = T_local + dx[-2]
          self.Re_guess = Re_local + dx[-1]

          # update viscosity so that time forward map defined correctly
          self.viscosity = 1. / self.Re_guess

          self._update_dX_dr()
          self._update_F()
          self._update_du_dx()
          self._update_du_dt()
          self._update_du_dRe()

          newt_new = la.norm(self.F)
          Delta /= 2.
          hook_count += 1
          if hook_info == 'stop': break # couldn't bound Delta
        print("# hooksteps:", hook_count)
      print("Current Newton residual: ", la.norm(self.F) / 
           self.norm_field(self.omega_guess))
      print("x-shift guess: ", self.a_guess)
      print("T guess: ", self.T_guess)
      print("Re guess: ", self.Re_guess)
      newt_res = newt_new
      res_history.append(newt_res / self.norm_field(self.omega_guess))
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
    po_guess.record_outcome(jnp.fft.rfftn(self.omega_guess), self.T_guess, self.a_guess, res_history, newt_count) 
    return po_guess, self.Re_guess
      
  def _timestep_A(
      self, 
      eta_w_T: Array
      ) -> Array:
    # TODO replace finite difference jacobian estimate with jacfwd
    eps_new = self.eps_fd * self.norm_field(self.omega_guess) / la.norm(eta_w_T[:self.Ndof])

    # should clean up this back and forth
    array_to_timestep = self.omega_guess.reshape((-1,)) + eps_new * eta_w_T[:self.Ndof]
    omega_eta_T = self._timestep_DNS(array_to_timestep.reshape(self.original_shape), 
                                     self.T_guess,
                                     self.viscosity)

    omega_to_shift = omega_eta_T - self.omega_T
    Aeta = ((1./eps_new) * 
            self.shift_reflect_fn(insp.x_shift(omega_to_shift, self.grid, self.a_guess)).reshape((-1,)) - 
            eta_w_T[:self.Ndof])

    Aeta += self.dSomegaT_dx.reshape((-1,)) * eta_w_T[-3]
    Aeta += self.dSomegaT_dT.reshape((-1,)) * eta_w_T[-2]
    Aeta += self.dSomegaT_dRe.reshape((-1,)) * eta_w_T[-1]

    # append the constraints 
    Aeta_w_x = np.append(Aeta, np.dot(self.domega0_dx.reshape((-1,)), eta_w_T[:self.Ndof])) 
    Aeta_w_x_T = np.append(Aeta_w_x, np.dot(self.domega0_dt.reshape((-1,)), eta_w_T[:self.Ndof]))
    Aeta_w_x_T_Re = np.append(Aeta_w_x_T, np.dot(self.dX_dr, eta_w_T))
    return Aeta_w_x_T_Re
 
  def _update_F(
      self
      ):
    self.omega_T = self._timestep_DNS(self.omega_guess, self.T_guess, self.viscosity)
    shifted_omega_T_arr = self.shift_reflect_fn(insp.x_shift(self.omega_T, 
                                                             self.grid, 
                                                             self.a_guess
                                                             )).reshape((-1,))
    omega_g_arr = self.omega_guess.reshape((-1,))
    
    # final term from guess -- notation follows Chandler & Kerswell, JFM 2013
    X = np.append(
      self.omega_guess.reshape((-1,)),
      [self.a_guess, self.T_guess, self.Re_guess]
    )

    N_X = (X - self.X_0).dot(self.dX_dr) - self.dr

    self.F = np.append(shifted_omega_T_arr - omega_g_arr, [0., 0., N_X]) # zero for shift, T rows

  def _update_du_dRe(
      self,
      delta_Re: float=1e-7
  ):
    perturbed_viscosity = 1. / (self.Re_guess + delta_Re)
    omega_T_larger_Re = self._timestep_DNS(self.omega_guess, 
                                           self.T_guess, 
                                           perturbed_viscosity)
    self.dSomegaT_dRe = (
      self.shift_reflect_fn(insp.x_shift(omega_T_larger_Re, 
                                         self.grid, 
                                         self.a_guess)) - 
      self.shift_reflect_fn(insp.x_shift(self.omega_T, 
                                         self.grid, 
                                         self.a_guess))
    ) / delta_Re
    self.viscosity = 1. / self.Re_guess

  def _compute_dr(
      self
      ):
    """ Compute (fixed) arclength between two given solutions """
    dX = self.X_0 - self.X__1
    self.dr = la.norm(dX)
    print("Physical arclength: ", self.dr)

  def _update_dX_dr(self):
    X = np.append(
      self.omega_guess.reshape((-1,)),
      [self.a_guess, self.T_guess, self.Re_guess]
    )
    self.dX_dr = (X - self.X__1) / (2 * self.dr)

  # domega_dt needs redefining despite inheritcance due to dependence on Re in evaluation
  def _compute_domega_dt(
      self, 
      omega_0: Array 
  ) -> Array:
    omega_rft = jnp.fft.rfftn(omega_0)
    return insp.rhs_equations(omega_rft, self.grid, self.Re_guess)
  
def write_out_soln_info(file_front: str, 
                        num: int,
                        upo: poGuessSpectral,
                        Re: float):
  meta_info = jnp.array([upo.T_out, upo.shift_out, upo.n_shift_reflects, Re])
  jnp.save(file_front + str(num) + '_array.npy', upo.omega_rft_out)
  jnp.save(file_front + str(num) + '_meta.npy', meta_info)

import numpy as np 
import scipy.linalg as la
from typing import Callable, Tuple
Array = np.ndarray

class krylovBasis:
  def __init__(
      self, 
      rhs_vec: Array, 
      max_size: int,
      data_type: str='float64'
  ):
    self.rhs_vec = rhs_vec
    self.N = self.rhs_vec.size 
    self.max_size = max_size

    self.basis = np.empty((self.N, self.max_size), dtype=data_type)
    self.hessenberg = np.zeros((self.max_size, self.max_size), dtype=data_type)

    self.n_basis_vec = 0

  def add_basis_vector(
      self, 
      A_operator: Callable[[Array], Array]
  ):
    """ ADD DESCRIPTION A_operator x = rhs_vec """
    if self.n_basis_vec == 0:
      self.basis[:,0] = self.rhs_vec / la.norm(self.rhs_vec)

    v = A_operator(self.basis[:, self.n_basis_vec])
    for j in range(self.n_basis_vec + 1):
      self.hessenberg[j, self.n_basis_vec] = np.dot(v.conj(), self.basis[:,j]) # 
      v -= self.hessenberg[j, self.n_basis_vec] * self.basis[:,j]
    self.hessenberg[self.n_basis_vec+1, self.n_basis_vec] = la.norm(v)

    self.basis[:, self.n_basis_vec+1] = v / self.hessenberg[self.n_basis_vec+1, self.n_basis_vec]
    self.n_basis_vec += 1 

  def svd_of_hessenberg(
      self, 
      n_trunc: int
  ) -> Tuple[Array]:
    """ Role of n_trunc """
    U, D, W_H = la.svd(self.hessenberg[:n_trunc+1,:n_trunc], full_matrices=True)
    return U, D, W_H

  def find_z_n(
      self, 
      U: Array, 
      D_mat: Array, 
      iter_count: int
  ) -> Array:
    e_1 = np.zeros(iter_count+1)
    e_1[0] = 1.
    z_n = np.dot(la.inv(D_mat[:iter_count, :iter_count]), np.dot(U[:,:iter_count].conj().T, e_1)) * la.norm(self.rhs_vec)
    return z_n

  def find_x(
      self, 
      iter_count: int
  ) -> Array:
    """ Return GMRES solution x of argmin||A_n x - b|| """ 
    U, D, W_H = self.svd_of_hessenberg(iter_count)
    D_mat = np.append(np.diag(D), [np.zeros(iter_count)], axis=0)
    z_n = self.find_z_n(U, D_mat, iter_count)

    y_n = np.dot(W_H.conj().T, z_n)  # coefficients of q_n (basis) in soln
    du_n = np.dot(self.basis[:,:iter_count], y_n) 
    return du_n

  def return_eigenvalues_hessenberg(
      self
  ) -> Array:
    reduced_hessenberg = self.hessenberg[:self.n_basis_vec, :self.n_basic_vec]
    return la.eig(reduced_hessenberg)

def gmres(
    A_operator: Callable[[Array], Array], 
    rhs_vec: Array, 
    gmres_tol: float=1e-3, 
    max_iter: int=100,
    data_type: str='float64'
) -> Tuple[krylovBasis, float, int]:
  res = 1.
  iter_count = 1
  basis = krylovBasis(rhs_vec, max_iter, data_type=data_type)
  while(res > gmres_tol) and (iter_count < max_iter):
    basis.add_basis_vector(A_operator)
    U, _, _ = basis.svd_of_hessenberg(iter_count)
    e_1 = np.zeros(iter_count+1)
    e_1[0] = 1.
    p_n1 = np.dot(U.conj().T, e_1) * la.norm(rhs_vec)

    res = abs(p_n1[iter_count]) / la.norm(p_n1)
    iter_count += 1
    #print("Current GMRES res: ", res, " and number of loops: ", iter_count-1)
  
  print("GMRES residual: ", res, "Iterations: ", iter_count-1)
  return basis, res, iter_count-1 

def hookstep(
    basis: krylovBasis, 
    Delta: float,
    data_type: str='float64'
) -> Tuple[Array, bool]:
  """ Some details from C and K on notation; as above  """ 
  U, D, W_H = basis.svd_of_hessenberg(basis.n_basis_vec) 
  e_1 = np.zeros(basis.n_basis_vec+1)
  e_1[0] = 1.
  p_n1 = np.dot(U.conj().T, e_1) * la.norm(basis.rhs_vec)

  def _update_zn(
      mu: float
  ) -> Array:
    z_n = np.zeros(p_n1.size-1, dtype=data_type)
    for i in range(basis.n_basis_vec):
      z_n[i] = p_n1[i] * D[i] / (mu + np.real(D[i]) ** 2)
    return z_n  

  def _return_updated_x(
      z_n: Array
  ) -> Array:
    y_n = np.dot(W_H.conj().T, z_n) 
    return np.dot(basis.basis[:,:basis.n_basis_vec], y_n) 

  mu_lower = 0.
  mu_upper = D[0]
  z_n = _update_zn(mu=mu_lower)
  if la.norm(z_n) < Delta:
    return _return_updated_x(z_n), True
  
  # find upper and lower bounds on mu
  while True:
    z_n = _update_zn(mu=mu_upper)
    if la.norm(z_n) < Delta: break
    mu_lower = mu_upper
    mu_upper *= 2
  
  # now bisect to obtain mu s.t. Delta/sqrt(2) < ||zn|| < sqrt(2)*Delta
  while True:
    mu_new = (mu_lower + mu_upper) / 2.
    z_n = _update_zn(mu=mu_new)
    if (la.norm(z_n) > Delta / (2.**0.5)) and (la.norm(z_n) < Delta * (2.**0.5)):
      break
    if la.norm(z_n) > Delta:
      mu_lower = mu_new
    else:
      mu_upper = mu_new
  return _return_updated_x(z_n), False


if __name__ == '__main__':
  # simple example from Trefethen and Bau
  n_it = 200
  A = 2 * np.eye(n_it, dtype='float') + (0.5 / (n_it ** 0.5)) * np.random.normal(size=(n_it, n_it))
  
  def A_mul(q):
    return A.dot(q)
  
  b = np.random.rand(n_it)
  kr_basis, res, iterations = gmres(A_mul, b, gmres_tol=1e-9, max_iter=50)
  kr_soln = kr_basis.find_x(iterations) 
  
  exact = la.inv(A).dot(b)
  
  error = la.norm(kr_soln - exact) / la.norm(exact)
  print(error)

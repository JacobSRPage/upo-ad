import jax.numpy as jnp

def downsample_rft_traj(vort_rft_traj: jnp.ndarray, N_downsample: int) -> jnp.ndarray:
  """Downsamples an array of Fourier coefficients obtained from jnp.fft.rfftn().

  Args:
    vort_rft_traj (jnp.ndarray): Input array of Fourier coefficients.
                                     First dimension represents time.
    N_downsample (int): Downsampling factor.

  Returns:
    jnp.ndarray: Downsampled array of Fourier coefficients.
  """  
  vort_shape = vort_rft_traj.shape

  # special treatment of final dimension
  vort_rft_traj = vort_rft_traj[..., :vort_shape[-1] // N_downsample + 1]

  # need to deal with front and back halves of rfft separately
  N_divide = 2 * N_downsample
  
  loc = 1
  for Nk in vort_shape[1:-1]:
    # deal with +ve wavenumbers first 
    front_slices = (
      [slice(0, vort_shape[0])] + (loc - 1) * [slice(0, s) for s in vort_shape[1:loc]] +
      [slice(0, Nk // N_divide)] + 
      (len(vort_shape) - loc - 1) * [slice(0, s) for s in vort_shape[loc+1:]]
    )

    idx = tuple(jnp.s_[s] for s in front_slices)
    vort_rft_front = vort_rft_traj[idx]

    # now deal with -ve wavenumbers 
    back_slices = (
      [slice(0, vort_shape[0])] + (loc - 1) * [slice(0, s) for s in vort_shape[1:loc]] +
      [slice(-Nk // N_divide, None)] +
      (len(vort_shape) - loc - 1) * [slice(0, s) for s in vort_shape[loc+1:]]
    )

    idx = tuple(jnp.s_[s] for s in back_slices)
    vort_rft_back = vort_rft_traj[idx]
    vort_rft_traj = jnp.concatenate([vort_rft_front, vort_rft_back], axis=loc)
    loc += 1

  return vort_rft_traj // N_downsample ** 2

def upsample_rft_traj(vort_rft_traj: jnp.ndarray, N_upsample: int) -> jnp.ndarray:
  """Upsamples an array of Fourier coefficients to a higher resolution grid.

  Args:
      vort_rft_traj (jnp.ndarray): Input array of Fourier coefficients.
      N_upsample (int): Upsampling factor.

  Returns:
      jnp.ndarray: Upsampled array of Fourier coefficients.
  """
  vort_shape = vort_rft_traj.shape

  loc = 1
  for Nk in vort_shape[1:-1]:
    # extract front (+ve wavenumbers)
    front_slices = (
      [slice(0, vort_shape[0])] + (loc - 1) * [slice(0, s) for s in vort_shape[1:loc]] +
      [slice(0, Nk // 2)] + 
      (len(vort_shape) - loc - 1) * [slice(0, s) for s in vort_shape[loc+1:]]
    )

    idx = tuple(jnp.s_[s] for s in front_slices)
    vort_rft_front = vort_rft_traj[idx]

    # extract back (-ve wavenumbers)
    back_slices = (
      [slice(0, vort_shape[0])] + (loc - 1) * [slice(0, s) for s in vort_shape[1:loc]] +
      [slice(-Nk // 2, None)] +
      (len(vort_shape) - loc - 1) * [slice(0, s) for s in vort_shape[loc+1:]]
    )

    idx = tuple(jnp.s_[s] for s in back_slices)
    vort_rft_back = vort_rft_traj[idx]

    # pad the relevant axis of vort_front with zeros
    padding_non_zero = [(0, Nk * (N_upsample - 1))]
    padding_instructions = ([(0, 0)] + 
                            (loc - 1) * [(0, 0)] + 
                            padding_non_zero + 
                            (len(vort_shape) - loc - 1) * [(0, 0)])
    vort_rft_front = jnp.pad(vort_rft_front, padding_instructions)

    vort_rft_traj = jnp.concatenate([vort_rft_front, vort_rft_back], axis=loc)
    loc += 1

  # special treatment of final dimension
  padding_instructions_final_dim = (0, (vort_shape[-1] - 1) * (N_upsample - 1))

  vort_rft_traj = jnp.pad(vort_rft_traj, 
                          [(0,0)] * (len(vort_shape) - 1) + [padding_instructions_final_dim])

  return vort_rft_traj * N_upsample ** 2


def rft_trajectory_to_physical(rft_traj: jnp.ndarray) -> jnp.ndarray:
  n_space_dim = len(rft_traj.shape[1:])
  return jnp.fft.irfftn(rft_traj, axes=tuple(range(1, n_space_dim + 1)))

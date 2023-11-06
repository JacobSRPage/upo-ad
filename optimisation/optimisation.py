import jax.numpy as jnp
import numpy as np

from opt_newt_jaxcfd.interact_jaxcfd import interact_jaxcfd_dtypes as glue

def iterate_optimizer_for_rpo(
    optimizer_triple, 
    initial_condition, 
    grid,
    loss,
    grad_loss,
    T_guess, 
    x_shift, 
    n_steps,
    write_out_step = None,
):
  opt_init, opt_update, get_params = optimizer_triple
  
  initial_field, offsets = glue.gv_tuple_to_jnp(initial_condition)
  original_shape = initial_field.shape
  original_bc = initial_condition[0].bc # assume is same for all components

  u_T_shift_init = glue.state_vector_extra_parameters(initial_condition, [T_guess, x_shift])
  opt_state = opt_init(u_T_shift_init)

  def step(
      step, 
      opt_state
  ):
    u_T_shift = get_params(opt_state)
    value = loss(glue.jnp_to_gv_tuple(u_T_shift[:-2].reshape(original_shape), offsets, grid, original_bc), 
                  u_T_shift[-2], u_T_shift[-1])
    grads = grad_loss(glue.jnp_to_gv_tuple(u_T_shift[:-2].reshape(original_shape), offsets, grid, original_bc), 
                  [u_T_shift[-2], u_T_shift[-1]])
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state
  
  loss_values = []
  T_values = []
  shift_values = []
  write_out_fields = []
  for j in range(n_steps):
    value, opt_state = step(j, opt_state)
    u_T_shift = get_params(opt_state)
    loss_values.append(value)
    T_values.append(u_T_shift[-2])
    shift_values.append(u_T_shift[-1])
    if j%5 == 0:
      print(j, value, u_T_shift[-2], u_T_shift[-1])
    if write_out_step is not None:
      if j%write_out_step == 0:
        write_out_fields.append(u_T_shift)
  if write_out_step is None:
    return (loss_values, T_values, shift_values), u_T_shift
  else:
    return (loss_values, T_values, shift_values), write_out_fields

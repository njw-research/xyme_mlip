import functools
import jax
import jax.numpy as jnp
import e3x
import flax.linen as nn
import chex


def prepare_single_sample(features, positions):
    # Determine the number of atoms in the sample
    n_nodes = positions.shape[0]
    # Compute the sparse pairwise indices for atom interactions
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_nodes)
    features = jnp.squeeze(features)
    chex.assert_rank(features, 1)
    return features, positions.reshape(-1, 3), dst_idx, src_idx, n_nodes


class MessagePassing(nn.Module):
  features: int = 32
  max_degree: int = 2
  num_iterations: int = 3
  num_basis_functions: int = 8
  cutoff: float = 5.0
  max_atomic_number: int = 118  # This is overkill for most applications.


  def energy(self, features, positions):

    features, positions, dst_idx, src_idx, n_nodes = prepare_single_sample(features, positions)

    # 1. Calculate displacement vectors.
    positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
    positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
    displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

    # 2. Expand displacement vectors in basis functions.
    basis = e3x.nn.basis(  # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
      displacements,
      num=self.num_basis_functions,
      max_degree=self.max_degree,
      radial_fn=e3x.nn.reciprocal_bernstein,
      cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
    )

    # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
    x = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1, features=self.features)(features)

    # 4. Perform iterations (message-passing + atom-wise refinement).
    for i in range(self.num_iterations):
      # Message-pass.
      if i == self.num_iterations-1:  # Final iteration.
        # Since we will only use scalar features after the final message-pass, we do not want to produce non-scalar
        # features for efficiency reasons.
        y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, dst_idx=dst_idx, src_idx=src_idx)
        # After the final message pass, we can safely throw away all non-scalar features.
        x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
      else:
        # In intermediate iterations, the message-pass should consider all possible coupling paths.
        y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
      y = e3x.nn.add(x, y)

      # Atom-wise refinement MLP.
      y = e3x.nn.Dense(self.features)(y)
      y = e3x.nn.silu(y)
      y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

      # Residual connection.
      x = e3x.nn.add(x, y)

    # 5. Predict atomic energies with an ordinary dense layer.
    element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
    atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)  # (..., Natoms, 1, 1, 1)
    atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # Squeeze last 3 dimensions.
    atomic_energies += element_bias[features]

    # 6. Sum atomic energies to obtain the total energy.
    energy = jnp.sum(atomic_energies)

    return -jnp.sum(energy), energy  # Forces are the negative gradient, hence the minus sign.

  @nn.compact
  def __call__(self, features, positions):
    # Define the energy and forces computation for a single sample
    def energy_and_forces_single(features, positions):
        energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
        (_, energy), forces = energy_and_forces(features, positions)
        return energy, forces

    # Check the shape of the positions array
    if len(positions.shape) == 2:
        # If positions are 2D (single sample), no vmap needed
        energy, forces = energy_and_forces_single(features, positions)
    elif len(positions.shape) == 3:
        # If positions are 3D (batched samples), apply vmap
        energy_and_forces_batched = jax.vmap(energy_and_forces_single, in_axes=(0, 0))
        energy, forces = energy_and_forces_batched(features, positions)
    else:
        raise ValueError("Unexpected shape for positions array. Expected rank 2 or 3.")

    return energy, forces

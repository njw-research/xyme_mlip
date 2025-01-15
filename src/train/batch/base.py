from typing import Tuple, Optional
import chex
import jax
import jax.numpy as jnp
Params = chex.ArrayTree

def setup_padded_reshaped_data(data: chex.ArrayTree,
                               interval_length: int,
                               reshape_axis=0) -> Tuple[chex.ArrayTree, chex.Array]:
    test_set_size = jax.tree_util.tree_flatten(data)[0][0].shape[0]
    chex.assert_tree_shape_prefix(data, (test_set_size, ))

    padding_amount = (interval_length - test_set_size % interval_length) % interval_length
    test_data_padded_size = test_set_size + padding_amount
    test_data_padded = jax.tree_map(
        lambda x: jnp.concatenate([x, jnp.zeros((padding_amount, *x.shape[1:]), dtype=x.dtype)], axis=0), data
    )
    mask = jnp.zeros(test_data_padded_size, dtype=int).at[jnp.arange(test_set_size)].set(1)


    if reshape_axis == 0:  # Used for pmap.
        test_data_reshaped, mask = jax.tree_map(
            lambda x: jnp.reshape(x, (interval_length, test_data_padded_size // interval_length, *x.shape[1:])),
            (test_data_padded, mask)
        )
    else:
        assert reshape_axis == 1  # for minibatching
        test_data_reshaped, mask = jax.tree_map(
            lambda x: jnp.reshape(x, (test_data_padded_size // interval_length, interval_length, *x.shape[1:])),
            (test_data_padded, mask)
        )
    return test_data_reshaped, mask


def maybe_masked_mean(array: chex.Array, mask: Optional[chex.Array]):
    chex.assert_rank(array, 1)
    if mask is None:
        return jnp.mean(array)
    else:
        chex.assert_equal_shape([array, mask])
        array = jnp.where(mask, array, jnp.zeros_like(array))
        divisor = jnp.sum(mask)
        divisor = jnp.where(divisor == 0, jnp.ones_like(divisor), divisor)  # Prevent nan when fully masked.
        return jnp.sum(array) / divisor

def maybe_masked_max(array: chex.Array, mask: Optional[chex.Array]):
    chex.assert_rank(array, 1)
    if mask is None:
        return jnp.max(array)
    else:
        chex.assert_equal_shape([array, mask])
        array = jnp.where(mask, array, jnp.zeros_like(array)-jnp.inf)
        return jnp.max(array)


def get_tree_leaf_norm_info(tree):
    """Returns metrics about contents of PyTree leaves.

    Args:
        tree (_type_): _description_

    Returns:
        _type_: _description_
    """
    norms = jax.tree_util.tree_map(jnp.linalg.norm, tree)
    norms = jnp.stack(jax.tree_util.tree_flatten(norms)[0])
    max_norm = jnp.max(norms)
    min_norm = jnp.min(norms)
    mean_norm = jnp.mean(norms)
    median_norm = jnp.median(norms)
    info = {}
    info.update(
        per_layer_max_norm=max_norm,
        per_layer_min_norm=min_norm,
        per_layer_mean_norm=mean_norm,
        per_layer_median_norm=median_norm,
    )
    return info


def batchify_array(data: chex.Array, batch_size: int):
    num_datapoints = get_leading_axis_tree(data, 1)[0]
    batch_size = min(batch_size, num_datapoints)
    x = data[: num_datapoints - num_datapoints % batch_size]
    return jnp.reshape(
        x,
        (-1, batch_size, *x.shape[1:]),
    )


def batchify_data(data: chex.ArrayTree, batch_size: int):
    return jax.tree_map(lambda x: batchify_array(x, batch_size), data)


def get_leading_axis_tree(tree: chex.ArrayTree, n_dims: int = 1):
    flat_tree, _ = jax.tree_util.tree_flatten(tree)
    leading_shape = flat_tree[0].shape[:n_dims]
    chex.assert_tree_shape_prefix(tree, leading_shape)
    return leading_shape


def get_shuffle_and_batchify_data_fn(train_data: chex.ArrayTree, batch_size: int):
    def shuffle_and_batchify_data(train_data_array, key):
        key, subkey = jax.random.split(key)
        permutted_train_data = jax.random.permutation(subkey, train_data_array, axis=0)
        batched_data = batchify_array(permutted_train_data, batch_size)
        return batched_data

    return lambda key: jax.tree_map(
        lambda x: shuffle_and_batchify_data(x, key), train_data
    )
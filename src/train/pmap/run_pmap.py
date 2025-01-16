import chex
import jax
import jax.numpy as jnp
from typing import Tuple
from absl import logging
from functools import partial

# from eacf.train.base import get_shuffle_and_batchify_data_fn
from src.train.batch.base import get_shuffle_and_batchify_data_fn
from src.utils.containers import TrainingState
from src.utils.pmap import get_from_first_device
from src.train.pmap.training_pmap import train_pmap
from src.train.steps.steps import training_step, validation_step


def run_pmap(
    opt, 
    loss_fn_partial, 
    samples_train, 
    samples_valid, 
    state, 
    start_epoch, 
    num_epochs, 
    save_checkpoint_path, 
    checkpointing_enabled, 
    checkpoint_update_freq, 
    batch_size, 
    n_devices, 
    pmap_axis_name, 
    data_rng_key_generator,
    output_keys
):
    
    # Check that num_valid >= n_devices * batch_size
    num_valid = samples_valid.positions.shape[0] * batch_size * n_devices 
    if num_valid < n_devices * batch_size:
        raise ValueError(f"Insufficient validation data: num_valid ({num_valid}) must be >= n_devices * batch_size "
                         f"({n_devices * batch_size}).")

    # Training step function with pmap
    training_step_fn = partial(training_step, optimizer=opt, loss_fn=loss_fn_partial, use_pmap=True)

    def step_function(state: TrainingState, x: chex.ArrayTree) -> Tuple[TrainingState, dict]:
        key, subkey = jax.random.split(state.key)
        params, opt_state, info = training_step_fn(state.params, x, state.opt_state, subkey)
        return TrainingState(params=params, opt_state=opt_state, key=key), info

    def update_fn(state: TrainingState) -> Tuple[TrainingState, dict]:
        batchify_data = get_shuffle_and_batchify_data_fn(samples_train, batch_size * n_devices)
        data_shuffle_key = next(data_rng_key_generator)  # Use separate key gen to avoid grabbing from state.
        batched_data = batchify_data(data_shuffle_key)
        
        for i in range(batched_data.positions.shape[0]):
            x = batched_data[i]
            # Reshape data for pmap
            x = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (n_devices, batch_size, *x.shape[1:])), x)
            state, info = jax.pmap(step_function, axis_name=pmap_axis_name)(state, x)
        return state, get_from_first_device(info, as_numpy=False)

    # Validation step function with pmap
    valid_step_fn = partial(validation_step, loss_fn=loss_fn_partial)

    def valid_step_function(state: TrainingState, x: chex.ArrayTree) -> dict:
        key, subkey = jax.random.split(state.key)
        info = valid_step_fn(state.params, x, subkey)
        return info

    def valid_fn(state: TrainingState) -> dict:
        batchify_data = get_shuffle_and_batchify_data_fn(samples_valid, batch_size * n_devices)
        data_shuffle_key = next(data_rng_key_generator)  # Use separate key gen to avoid grabbing from state.
        batched_data = batchify_data(data_shuffle_key)

        for i in range(batched_data.positions.shape[0]):
            x = batched_data[i]
            # Reshape data for pmap
            x = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (n_devices, batch_size, *x.shape[1:])), x)
            info = jax.pmap(valid_step_function, axis_name=pmap_axis_name)(state, x)
        return get_from_first_device(info, as_numpy=False)

    # Start training with pmap
    state = train_pmap(state=state, 
               start_epoch=start_epoch, 
               num_epochs=num_epochs, 
               update_fn=update_fn, 
               valid_fn=valid_fn, 
               checkpointing_enabled=checkpointing_enabled, 
               checkpoint_dir=save_checkpoint_path,
               checkpoint_update_freq=checkpoint_update_freq,
               output_keys=output_keys,)

    logging.info("Training completed. Well done!")

    return state

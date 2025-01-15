import chex
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from optax._src.transform import ScaleByAdamState
from tqdm import tqdm
from typing import Callable, Tuple, Optional, List
from absl import logging

from src.train.batch.base import get_shuffle_and_batchify_data_fn
from src.utils.containers import TrainingState
from src.utils.checkpoint.checkpoint_state import save_state
from src.utils.containers import Graph

def create_scan_fn(
    partial_training_step: Callable[
        [hk.Params, Graph, chex.ArrayTree, chex.PRNGKey],
        Tuple[hk.Params, optax.OptState, dict],
    ],
    last_iter_info_only: Optional[bool] = False,
) -> Callable[
    [Tuple[chex.ArrayTree, optax.OptState, chex.PRNGKey], Graph],
    Tuple[Tuple[chex.ArrayTree, optax.OptState, chex.PRNGKey], dict],
]:
    def scan_fn(
        carry: Tuple[chex.ArrayTree, optax.OptState, chex.PRNGKey], xs: Graph
    ) -> Tuple[Tuple[chex.ArrayTree, optax.OptState, chex.PRNGKey], dict]:
        params, opt_state, key = carry
        key, subkey = jax.random.split(key)
        params, opt_state, info = partial_training_step(params, xs, opt_state, subkey)
        return (params, opt_state, key), info

    return scan_fn


def create_scan_epoch_fn(
    training_step: Callable[
        [hk.Params, Graph, chex.ArrayTree, chex.PRNGKey],
        Tuple[hk.Params, optax.OptState, dict],
    ],
    data,
    batch_size: int,
    last_iter_info_only: Optional[bool] = False,
):
    scan_fn = create_scan_fn(training_step, last_iter_info_only)

    shuffle_and_batchify_data = get_shuffle_and_batchify_data_fn(data, batch_size)

    def scan_epoch(params, opt_state, key):
        batched_data = shuffle_and_batchify_data(key)

        (params, opt_state, key), info = jax.lax.scan(
            scan_fn, (params, opt_state, key), batched_data, unroll=1
        )

        return params, opt_state, key, info

    return jax.jit(scan_epoch)

def create_validation_scan_fn(
    validation_step: Callable[
        [hk.Params, Graph, chex.PRNGKey], dict
    ],
    last_iter_info_only: Optional[bool] = False,
) -> Callable[
    [Tuple[chex.ArrayTree, chex.PRNGKey], Graph],
    Tuple[Tuple[chex.ArrayTree, chex.PRNGKey], dict],
]:
    def scan_fn(
        carry: Tuple[chex.ArrayTree, chex.PRNGKey], xs: Graph
    ) -> Tuple[Tuple[chex.ArrayTree, chex.PRNGKey], dict]:
        params, key = carry
        key, subkey = jax.random.split(key)
        info = validation_step(params, xs, subkey)

        return (params, key), info

    return scan_fn


def create_validation_epoch_fn(
    validation_step: Callable[
        [hk.Params, Graph, chex.PRNGKey], dict
    ],
    data,
    batch_size: int,
    last_iter_info_only: Optional[bool] = False,
):
    scan_fn = create_validation_scan_fn(validation_step, last_iter_info_only)

    shuffle_and_batchify_data = get_shuffle_and_batchify_data_fn(data, batch_size)

    def scan_epoch(params, key):
        batched_data = shuffle_and_batchify_data(key)

        (params, key), info = jax.lax.scan(
            scan_fn, (params, key), batched_data, unroll=1
        )

        return info

    return jax.jit(scan_epoch)

def train_scan(
        state: TrainingState, 
        start_epoch: int, 
        num_epochs: int, 
        update_fn: Callable[[chex.PRNGKey, hk.Params, chex.Array, Tuple[ScaleByAdamState, chex.Array]], Tuple[hk.Params, Tuple[ScaleByAdamState, chex.Array], chex.Array]], 
        valid_fn: Callable[[chex.PRNGKey, hk.Params, chex.Array], chex.Array], 
        checkpointing_enabled: bool, 
        checkpoint_dir: str,
        checkpoint_update_freq: int,
        output_keys: List[str] = None
    ) -> None:
    """
    Train the model using the specified parameters.

    Args:
        state (TrainingState): Initial model params, state and key.
        start_epoch (int): The epoch to start training from.
        num_epochs (int): The total number of epochs for training.
        update_fn (Callable): The update function for training.
        valid_fn (Callable): The validation function.
        checkpointing_enabled (bool): Flag to enable checkpointing.
        checkpoint_dir (str): Directory to save checkpoints.
        checkpoint_update_freq (int): Frequency of updates.
        output_keys (List[str]): Keys in the info dictionary to include in the output.

    Returns:
        None
    """

    # Initialize progress bar for tracking training progress
    pbar = tqdm(total=num_epochs, initial=start_epoch, desc="Training", unit="epoch")

    # Training loop with tqdm for progress tracking
    for epoch in range(start_epoch, num_epochs):
        # Training step using lax.scan
        state, info = update_fn(state)

        # Checkpointing and validation 
        if (epoch + 1) % checkpoint_update_freq == 0:
            # Validation step using lax.scan
            valid_info = valid_fn(state)

            # Create output string for logging based on specified output keys
            if output_keys:
                info_strings = [
                    f"{key}: {jnp.mean(info[key]):.4f}" if 'loss' in key or 'mae' in key else f"{key}: {jnp.mean(info[key])}"
                    for key in output_keys if key in info
                ]
                valid_info_strings = [
                    f"valid {key}: {jnp.mean(valid_info[key]):.4f}" if 'loss' in key or 'mae' in key else f"valid {key}: {jnp.mean(valid_info[key])}"
                    for key in output_keys if key in valid_info
                ]
            else:
                # Default keys if none provided
                info_strings = [
                    f"grad norm {jnp.mean(info['grad_norm'])}", 
                    f"train loss: {jnp.mean(info['loss']):.4f}"
                ]
                valid_info_strings = [
                    f"valid loss: {jnp.mean(valid_info['loss']):.4f}"
                ]

            output = f"Epoch {epoch + 1} | " + " | ".join(info_strings + valid_info_strings)
            pbar.set_description(output)  # Update progress bar description
            pbar.update(checkpoint_update_freq)  # Update
            logging.info(output)  # Log the output
            if checkpointing_enabled:
                save_state(checkpoint_dir, state, epoch)

    pbar.close()
    return state

# def train_scan(
#         state: TrainingState, 
#         start_epoch: int, 
#         num_epochs: int, 
#         update_fn: Callable[[chex.PRNGKey, hk.Params, chex.Array, Tuple[ScaleByAdamState, chex.Array]], Tuple[hk.Params, Tuple[ScaleByAdamState, chex.Array], chex.Array]], 
#         valid_fn: Callable[[chex.PRNGKey, hk.Params, chex.Array], chex.Array], 
#         checkpointing_enabled: bool, 
#         checkpoint_dir: str,
#         checkpoint_update_freq: int
#     ) -> None:
#     """
#     Train the model using the specified parameters.

#     Args:
#         state (TrainingState): Initial model params, state and key.
#         start_epoch (int): The epoch to start training from.
#         num_epochs (int): The total number of epochs for training.
#         update_fn (Callable): The update function for training.
#         valid__fn (Callable): The validation function.
#         checkpointing_enabled (bool): Flag to enable checkpointing.
#         checkpoint_dir (str): Directory to save checkpoints.
#         checkpoint_update_freq (int): Frequancy of updates.

#     Returns:
#         None
#     """


#     # Initialize progress bar for tracking training progress
#     pbar = tqdm(total=num_epochs, initial=start_epoch, desc="Training", unit="epoch")

#     # Training loop with tqdm for progress tracking
#     for epoch in range(start_epoch, num_epochs):
#         # Training step using lax.scan
#         state, info = update_fn(state)

#         # Checkpointing and validation 
#         if (epoch + 1) % checkpoint_update_freq == 0:
#             # Vadlidation step using lax.scan
#             valid_info = valid_fn(state)

#             # Create output string for logging
#             output = f"Epoch {epoch + 1} | grad norm {jnp.mean(info['grad_norm'])} | train loss: {jnp.mean(info['loss']):.4f}, train aux-loss: {jnp.mean(info['aux_loss']):.4f} | valid loss: {jnp.mean(valid_info['loss']):.4f}, valid aux-loss: {jnp.mean(valid_info['aux_loss']):.4f}"     
#             pbar.set_description(output)  # Update progress bar description
#             pbar.update(checkpoint_update_freq)  # Update
#             logging.info(output)  # Log the output
#             if checkpointing_enabled:
#                 save_state(checkpoint_dir, state, epoch)

#     pbar.close()

#     return state
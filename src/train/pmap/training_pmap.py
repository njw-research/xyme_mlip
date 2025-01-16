import chex
import jax.numpy as jnp
import haiku as hk
from optax._src.transform import ScaleByAdamState
from absl import logging
from tqdm import tqdm
from typing import Callable, Tuple, List
from src.utils.containers import TrainingState
from src.utils.checkpoint.checkpoint_state import save_state


def train_pmap(
        state: TrainingState, 
        start_epoch: int, 
        num_epochs: int, 
        update_fn: Callable[[chex.PRNGKey, hk.Params, chex.Array, Tuple[ScaleByAdamState, chex.Array]], Tuple[hk.Params, Tuple[ScaleByAdamState, chex.Array], chex.Array]], 
        valid_fn: Callable[[chex.PRNGKey, hk.Params, chex.Array], chex.Array], 
        checkpointing_enabled: bool, 
        checkpoint_dir: str,
        checkpoint_update_freq:int,
        output_keys: List[str] = None
    ) -> None:
    """
    Train the model using the specified parameters.

    Args:
        state (TrainingState): Initial model params, state and key.
        start_epoch (int): The epoch to start training from.
        num_epochs (int): The total number of epochs for training.
        update_fn (Callable): The update function for training.
        valid__fn (Callable): The validation function.
        checkpointing_enabled (bool): Flag to enable checkpointing.
        checkpoint_dir (str): Directory to save checkpoints.
        checkpoint_update_freq (int): Frequancy of updates.

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
            # Vadlidation step using lax.scan
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

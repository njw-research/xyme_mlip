import os
import pickle
import jax
import re
import chex
import optax
import jax.numpy as jnp
from src.utils.containers import TrainingState
from typing import Callable, Tuple, Optional

def restore_or_initialize_state(
    key: chex.PRNGKey, 
    model: Callable, 
    opt: optax.OptState, 
    data: chex.Array, 
    path: str, 
    n_devices: int, 
    checkpointing_enabled=True
) -> Tuple[TrainingState, int]:
    """Restore the state from a checkpoint if enabled, otherwise initialize the state.

    Args:
        key (chex.PRNGKey): Random key for initialization.
        flow (AugmentedFlow): The flow model.
        opt (optax.OptState): The optimizer state.
        data (chex.Array): The data used for initialization.
        path (str): Path to the checkpoint directory.
        n_devices (int): Number of devices for distributed training.
        checkpointing_enabled (bool): Flag to enable/disable checkpointing.

    Returns:
        state (TrainingState): The initialized or restored training state.
        epoch (int): The training epoch (0 if initializing).

    Raises:
        FileNotFoundError: If state restoration fails or no valid checkpoint is found.
    """
    if checkpointing_enabled and path:
        try:
            print(f"Attempting to restore state from checkpoint at {path}...")
            state, epoch = restore_state(path)
            return state, epoch +1 # Return if restoration is successful
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"Error encountered while restoring state: {e}")

    print("Checkpointing disabled or restoration failed. Initializing state...")
    epoch = 0

    def init_fn_single_devices(common_key: chex.PRNGKey, per_device_key: chex.PRNGKey) -> TrainingState:
        """Initialize the state for a single device."""
        params = model.init(common_key, data[0])
        opt_state = opt.init(params)
        return TrainingState(params, opt_state, per_device_key)

    def init_fn(key: chex.PRNGKey) -> TrainingState:
        common_key, per_device_key = jax.random.split(key)
        common_keys = jnp.repeat(common_key[None, ...], n_devices, axis=0)
        per_device_keys = jax.random.split(per_device_key, n_devices)
        init_state = jax.pmap(init_fn_single_devices)(common_keys, per_device_keys)
        chex.assert_trees_all_equal(
            jax.tree_util.tree_map(lambda x: x[0], init_state.params),
            jax.tree_util.tree_map(lambda x: x[1], init_state.params)
        )
        assert (init_state.key[0] != init_state.key[1]).all()  # Check rng per state is different.
        return init_state

    state = init_fn(key) if n_devices > 1 else init_fn_single_devices(*jax.random.split(key))
    print("State initalized.")
    return state, epoch

def condense_state_to_single_device(init_state: TrainingState) -> TrainingState:
    """Condense the state from multiple devices to a single device."""
    params_single = jax.tree_util.tree_map(lambda x: x[0], init_state.params)
    opt_state_single = jax.tree_util.tree_map(lambda x: x[0], init_state.opt_state)
    key_single = init_state.key[0]  # Use the first key
    return TrainingState(params_single, opt_state_single, key_single)

def save_state(ckpt_dir: str, state: TrainingState, epoch: int) -> None:
    """Save model parameters and state.
    
    Args:
        ckpt_dir (str): Directory to save the checkpoint.
        state (TrainingState): The current training state.
        epoch (int): The current epoch.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_file = os.path.join(ckpt_dir, f'model_checkpoint_{epoch}.pkl')
    
    checkpoint_data = {
        "state": state,
        "epoch": epoch,
    }
    
    with open(ckpt_file, "wb") as f:
        pickle.dump(checkpoint_data, f)

def restore_state(ckpt_dir: str) -> Tuple[dict, int]:
    """Restore model parameters and state from the checkpoint directory.
    
    Args:
        ckpt_dir (str): Directory to restore from.
    
    Raises:
        FileNotFoundError: If the checkpoint directory does not exist or no checkpoint files are found.
    
    Returns:
        state (dict): The restored model state.
        epoch (int): The epoch at which the checkpoint was saved.
    """
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory '{ckpt_dir}' does not exist.")
    
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith('model_checkpoint')]
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in directory '{ckpt_dir}'.")

    checkpoints.sort(key=lambda f: int(re.search(r'model_checkpoint_(\d+)\.pkl', f).group(1)))

    latest_checkpoint = checkpoints[-1]  # Get the latest checkpoint
    ckpt_file = os.path.join(ckpt_dir, latest_checkpoint)

    # Load the checkpoint data
    with open(ckpt_file, "rb") as f:
        checkpoint_data = pickle.load(f)
    
    return checkpoint_data["state"], checkpoint_data["epoch"]
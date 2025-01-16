import os
import jax
import haiku as hk

from src.utils.checkpoint.checkpoint_utils import find_highest_train_directory
from src.utils.data.load_data import load_and_prepare_datasets

def get_config(data_dir: str, 
               dataset: str, 
               checkpoint_dir: str,
               num_train: int, 
               num_valid: int
               ):

    # Initialize random number generator (RNG) keys for reproducibility
    rng_key, data_key, train_key, init_key = jax.random.split(jax.random.PRNGKey(0), 4)

    # Get available JAX devices (e.g., GPUs)
    devices = jax.devices()
    n_devices = len(devices)
    pmap_axis_name = 'data'
    print("Current JAX device:", devices)
    print("Number of devices found: ", n_devices)

    # Find the highest training directory for resuming training
    script_dir = os.path.dirname(os.path.abspath(__file__))
    highest_train_dir = find_highest_train_directory(script_dir)
    restore_checkpoint_dir = os.path.join(highest_train_dir, checkpoint_dir) if highest_train_dir else None

    # Data RNG key sequence for shuffling
    data_rng_key_generator = hk.PRNGSequence(42)

    # Load training and validation datasets
    samples_train, samples_valid, atomic_numbers, n_nodes = load_and_prepare_datasets(
        script_dir, 
        data_key, 
        data_dir,  # Passed from arguments
        dataset,   # Passed from arguments
        num_train, # Passed from arguments
        num_valid  # Passed from arguments
    )

    config_dict = {
        'rng_key': rng_key,
        'data_key': data_key,
        'train_key': train_key,
        'init_key': init_key,
        'devices': devices,
        'n_devices': n_devices,
        'pmap_axis_name': pmap_axis_name,
        'highest_train_dir': highest_train_dir,
        'restore_checkpoint_dir': restore_checkpoint_dir,
        'data_rng_key_generator': data_rng_key_generator,
        'samples_train': samples_train,
        'samples_valid': samples_valid,
        'atomic_numbers': atomic_numbers
        }

    return  config_dict, n_devices

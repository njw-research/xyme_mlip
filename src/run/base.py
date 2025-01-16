# Import standard libraries
import os
import jax
import haiku as hk
from functools import partial
from absl import logging, flags
import haiku.experimental.flax as hkflax
import sys
import chex
from jax.flatten_util import ravel_pytree

# Import custom libraries
from src.utils.checkpoint.checkpoint_utils import find_highest_train_directory
from src.utils.data.load_data import load_and_prepare_datasets
from src.utils.checkpoint.checkpoint_state import restore_or_initialize_state
from src.utils.containers import Graph
from src.train.optimizer.optimizer import get_optimizer
from src.mlip.message_passing import MessagePassing
from src.train.loss.loss import loss_fn_apply
from src.utils.containers import TrainingState
from src.utils.setup_logging import setup_logging
from src.utils.checkpoint.checkpoint_utils import create_training_directory

def param_count(x: chex.ArrayTree) -> int:
    """Count the number of parameters in a PyTree of parameters."""
    return ravel_pytree(x)[0].shape[0]

# Initialize training process and directories
def initialize_checkpoint_path(state: TrainingState, 
                        start_epoch: int,
                        num_epochs: int, 
                        script_dir: str, 
                        logging_filename: str, 
                        checkpoint_dir: str):
    
    if start_epoch >= num_epochs:
        print(f"Training already completed. Exiting. start_epoch: {start_epoch}, num_epochs: {num_epochs}")
        sys.exit(0)

    # Create directory for saving training checkpoints and set up logging
    train_dir = create_training_directory(script_dir)
    save_checkpoint_path = os.path.join(train_dir, checkpoint_dir)
    setup_logging(train_dir, logging_filename)
    logging.info(f"Starting epoch: {start_epoch}, params count: {param_count(state.params)}")

    return save_checkpoint_path


FLAGS = flags.FlagValues()

def get_config(
        script_dir: str,
        data_dir: str, 
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

def initalize_run(FLAGS):
    # Get configuration dictionary and number of devices
    config, n_devices = get_config(
        FLAGS.script_dir,
        FLAGS.data_dir,
        FLAGS.dataset,
        FLAGS.checkpoint_dir,
        FLAGS.num_train,
        FLAGS.num_valid
        )

    # Instantiate the message-passing model
    message_passing = MessagePassing(
        features=FLAGS.num_features,
        max_degree=FLAGS.max_degree,
        num_iterations=FLAGS.num_iterations,
        num_basis_functions=FLAGS.num_basis_functions,
        cutoff=FLAGS.cutoff,
        max_atomic_number=FLAGS.max_atomic_number
    )

    # Define energy and force computation with Haiku model transformation
    @hk.without_apply_rng
    @hk.transform
    def energy_and_forces(graph: Graph):
        mod = hkflax.lift(message_passing, name='e3x_mlip')
        return mod(graph.features, graph.positions)

    # Initialize the optimizer
    opt = get_optimizer(FLAGS.learning_rate, FLAGS.gradient_clipping)

    # Restore or initialize the model state
    state, start_epoch = restore_or_initialize_state(
        key=config['init_key'],
        model=energy_and_forces,
        opt=opt,
        data=config['samples_train'],
        path=config['restore_checkpoint_dir'],
        n_devices=config['n_devices']
    )

    # Set up paths and logging for training process
    save_checkpoint_path = initialize_checkpoint_path(
        state=state,
        start_epoch=start_epoch,
        num_epochs=FLAGS.num_epochs,
        script_dir=FLAGS.script_dir,
        logging_filename=FLAGS.logging_filename,
        checkpoint_dir=FLAGS.checkpoint_dir
    )

    # Partial loss function with force weighting
    loss_fn_partial = partial(
        loss_fn_apply,
        model_apply=energy_and_forces.apply,
        forces_weight=FLAGS.forces_weight,
    )

    return config, n_devices, state, start_epoch, save_checkpoint_path, opt, loss_fn_partial

# class Run_Functions:
#     def __init__(self., FLAGS):
#         self.FLAGS = FLAGS
#         self.config = None
#         self.n_devices = None
#         self.message_passing = None
#         self.energy_and_forces = None
#         self.opt = None
#         self.state = None
#         self.start_epoch = None
#         self.save_checkpoint_path = None
#         self.loss_fn_partial = None
#         self.get_config()
#         self.instantiate_model()
#         self.define_model()
#         self.init_optimizer()
#         self.restore_or_initialize_state()
#         self.initialize_training()

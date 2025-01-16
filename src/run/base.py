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
    """Count the number of parameters in a PyTree of parameters.
    
    Args:
        x (chex.ArrayTree): A PyTree representing the parameters of a model, typically created by a neural network library like Haiku or Flax.
    
    Returns:
        int: The total number of parameters in the PyTree. This is determined by raveling the tree and counting the elements of the resulting array.
    """
    return ravel_pytree(x)[0].shape[0]

# Initialize training process and directories
def initialize_checkpoint_path(state: TrainingState, 
                        start_epoch: int,
                        num_epochs: int, 
                        script_dir: str, 
                        logging_filename: str, 
                        checkpoint_dir: str):
    """Initialize the path for saving checkpoints during training.
    
    Args:
        state (TrainingState): The current state of the training process, including parameters and other relevant data.
        start_epoch (int): The epoch at which training is starting or resuming from.
        num_epochs (int): The total number of epochs for the training process.
        script_dir (str): The directory where the script is located, used for creating subdirectories.
        logging_filename (str): The name of the file where logs will be recorded during training.
        checkpoint_dir (str): The name of the subdirectory where checkpoints will be saved.
    
    Returns:
        str: The full path to the directory where checkpoints will be saved.
    """
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
    """Get the configuration for initializing and running a training process.
    
    Args:
        script_dir (str): The directory where the script is located.
        data_dir (str): The directory containing the dataset files.
        dataset (str): The name of the dataset to be used for training and validation.
        checkpoint_dir (str): The name of the subdirectory where checkpoints will be saved.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.
    
    Returns:
        tuple: A tuple containing the configuration dictionary and the number of devices found.
    """
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

FLAGS = flags.FlagValues()

class RunInitializer:
    def __init__(self, FLAGS):
        """Initialize the RunInitializer class with flags.
        
        Args:
            FLAGS: The flag values object containing configuration parameters for the training process.
        """
        self.FLAGS = FLAGS
        self._initialize_components()

    def _get_config(self):
        """Get the configuration for initializing and running a training process.
        
        Returns:
            tuple: A tuple containing the configuration dictionary and the number of devices found.
        """
        return get_config(
            self.FLAGS.script_dir,
            self.FLAGS.data_dir,
            self.FLAGS.dataset,
            self.FLAGS.checkpoint_dir,
            self.FLAGS.num_train,
            self.FLAGS.num_valid
        )

    def _instantiate_message_passing(self):
        """Instantiate the message passing module for processing graph data.
        
        Returns:
            MessagePassing: An instance of the MessagePassing class configured with specific parameters.
        """
        return MessagePassing(
            features=self.FLAGS.num_features,
            max_degree=self.FLAGS.max_degree,
            num_iterations=self.FLAGS.num_iterations,
            num_basis_functions=self.FLAGS.num_basis_functions,
            cutoff=self.FLAGS.cutoff,
            max_atomic_number=self.FLAGS.max_atomic_number
        )

    def _define_energy_and_forces(self):
        """Define the energy and forces computation using Haiku and Flax.
        
        Returns:
            function: A callable that computes energy and forces for a given graph.
        """
        message_passing = self._instantiate_message_passing()
        @hk.without_apply_rng
        @hk.transform
        def energy_and_forces(graph: Graph):
            mod = hkflax.lift(message_passing, name='e3x_mlip')
            return mod(graph.features, graph.positions)
        return energy_and_forces

    def _initialize_optimizer(self):
        """Initialize the optimizer for training the model.
        
        Returns:
            Optimizer: An instance of an optimizer configured with specific learning rate and gradient clipping.
        """
        return get_optimizer(self.FLAGS.learning_rate, self.FLAGS.gradient_clipping)

    def _restore_or_initialize_state(self, config):
        """Restore or initialize the state for training.
        
        Args:
            config (dict): The configuration dictionary containing necessary data for initialization.
        
        Returns:
            tuple: A tuple containing the restored or initialized state and the start epoch.
        """
        return restore_or_initialize_state(
            key=config['init_key'],
            model=self._define_energy_and_forces(),
            opt=self._initialize_optimizer(),
            data=config['samples_train'],
            path=config['restore_checkpoint_dir'],
            n_devices=config['n_devices']
        )

    def _initialize_checkpoint_path(self, state, start_epoch):
        """Initialize the path for saving checkpoints during training.
        
        Args:
            state (TrainingState): The current state of the training process, including parameters and other relevant data.
            start_epoch (int): The epoch at which training is starting or resuming from.
        
        Returns:
            str: The full path to the directory where checkpoints will be saved.
        """
        return initialize_checkpoint_path(
            state=state,
            start_epoch=start_epoch,
            num_epochs=self.FLAGS.num_epochs,
            script_dir=self.FLAGS.script_dir,
            logging_filename=self.FLAGS.logging_filename,
            checkpoint_dir=self.FLAGS.checkpoint_dir
        )

    def _partial_loss_fn(self):
        """Partial function to compute the loss for training.
        
        Returns:
            partial: A partially applied function to compute the loss using specific arguments.
        """
        return partial(
            loss_fn_apply,
            model_apply=self._define_energy_and_forces().apply,
            forces_weight=self.FLAGS.forces_weight,
        )

    def _initialize_components(self):
        """Initialize all necessary components for the training process.
        
        This includes configuring the data, initializing the optimizer and model, and setting up checkpointing.
        """
        config, n_devices = self._get_config()
        state, start_epoch = self._restore_or_initialize_state(config)
        save_checkpoint_path = self._initialize_checkpoint_path(state, start_epoch)
        opt = self._initialize_optimizer()
        loss_fn_partial = self._partial_loss_fn()
        self.config = config
        self.n_devices = n_devices
        self.state = state
        self.start_epoch = start_epoch
        self.save_checkpoint_path = save_checkpoint_path
        self.opt = opt
        self.loss_fn_partial = loss_fn_partial


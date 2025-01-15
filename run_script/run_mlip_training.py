# Import standard libraries
import sys
import os
import jax
import haiku as hk
import optax
from functools import partial
from absl import logging, app, flags
import haiku.experimental.flax as hkflax

from src.utils.base import param_count, initialize_training
from src.utils.setup_logging import setup_logging
from src.utils.checkpoint.checkpoint_utils import find_highest_train_directory
from src.utils.checkpoint.checkpoint_state import restore_or_initialize_state
from src.utils.data.load_data import load_and_prepare_datasets
from src.utils.containers import Graph
from src.train.optimizer.optimizer import OptimizerConfig
from src.train.scan.run_scan import run_scan
from src.train.pmap.run_pmap import run_pmap
from src.mlip.message_passing import MessagePassing
from src.train.loss.loss import loss_fn_apply


# Define flags 
FLAGS = flags.FLAGS

# Training paramaters 
flags.DEFINE_integer('num_epochs', 100, 'Number of training epochs.')
flags.DEFINE_integer('num_train', 64, 'Number of training samples.')
flags.DEFINE_integer('num_valid', 8, 'Number of validation samples.')
flags.DEFINE_integer('batch_size', 4, 'Batch size for training.')
flags.DEFINE_float('forces_weight', 10.0, "Weight on forces loss.")
flags.DEFINE_bool('last_iter_info_only', False, "Only use last batch for info.")
flags.DEFINE_list('output_keys', ["grad_norm", "energy_mae", "forces_mae"], 'Output keys for logging.')

# Checkpointing and data
flags.DEFINE_string('training_output_dir', './train_/', 'Directory for saving all model outputs.')
flags.DEFINE_boolean('checkpointing_enabled', True, 'Enable checkpointing.')
flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'Directory for saving model checkpoints.')
flags.DEFINE_integer('checkpoint_update_freq', 10, 'Frequency of checkpoint updating')
flags.DEFINE_string('logging_filename', './training.log', 'Log file for training progress.')
flags.DEFINE_string('data_dir', './data/', 'Directory for datasets.')
flags.DEFINE_string('dataset', 'md17_ethanol.npz', 'Name of the dataset file.')  #md17_ethanol.npz

# Dynamic optimization parmaters 
flags.DEFINE_float('init_lr', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('peak_lr', 2e-3, 'Peak learning rate for the schedule.')
flags.DEFINE_float('end_lr', 1e-4, 'End learning rate for the schedule.')
flags.DEFINE_string('optimizer_name', 'adam', 'Optimizer to use for training.')
flags.DEFINE_bool('use_schedule', False, 'Whether to use a learning rate schedule.')
flags.DEFINE_integer('warmup_n_epoch', 30, 'Number of epochs for learning rate warmup.')
flags.DEFINE_float('max_global_norm', None, 'Maximum global norm for gradient clipping (None for no clipping).')
flags.DEFINE_float('max_param_grad', None, 'Maximum parameter gradient (None for no gradient clipping).')
flags.DEFINE_bool('dynamic_grad_ignore_and_clip', True, 'Whether to ignore or clip gradients dynamically.')
flags.DEFINE_float('dynamic_grad_ignore_factor', 5., 'Factor above which to fully ignore the gradient.')
flags.DEFINE_float('dynamic_grad_norm_factor', 2., 'Factor above which to clip the gradient norm.')
flags.DEFINE_integer('dynamic_grad_norm_window', 20, 'Window size to track median gradient norm over.')

# Model hyperparameters
flags.DEFINE_integer('num_features', 8, 'Number of features')
flags.DEFINE_integer('max_degree', 2, 'Maximum degree')
flags.DEFINE_integer('num_iterations', 3, 'Number of iterations')
flags.DEFINE_integer('num_basis_functions', 16, 'Number of basis functions')
flags.DEFINE_float('cutoff', 5.0, 'Cutoff distance')
flags.DEFINE_integer('max_atomic_number', 9, 'Maximum atomic number')

def preface():
    global rng_key, data_key, train_key, init_key, devices, n_devices, pmap_axis_name
    global highest_train_dir, restore_checkpoint_dir, data_rng_key_generator, samples_train, samples_valid, atomic_numbers, n_nodes, optimizer_config
    
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
    restore_checkpoint_dir = os.path.join(highest_train_dir, FLAGS.checkpoint_dir) if highest_train_dir else None

    # Data RNG key sequence for shuffling
    data_rng_key_generator = hk.PRNGSequence(42)

    # Load training and validation datasets
    samples_train, samples_valid, atomic_numbers, n_nodes = load_and_prepare_datasets(
        script_dir, 
        data_key, 
        FLAGS.data_dir, 
        FLAGS.dataset, 
        FLAGS.num_train, 
        FLAGS.num_valid
    )

    # Creating OptimizerConfig instance using flag values
    optimizer_config = OptimizerConfig(
        init_lr=FLAGS.init_lr,
        end_lr=FLAGS.end_lr,
        dynamic_grad_ignore_and_clip=True
    )

# # Initialize training process and directories
# def initialize_training(start_epoch, state, script_dir):
#     if start_epoch >= FLAGS.num_epochs:
#         print(f"Training already completed. Exiting. start_epoch: {start_epoch}, num_epochs: {FLAGS.num_epochs}")
#         sys.exit(0)

#     # Create directory for saving training checkpoints and set up logging
#     train_dir = create_training_directory(script_dir)
#     save_checkpoint_path = os.path.join(train_dir, FLAGS.checkpoint_dir)
#     setup_logging(train_dir, FLAGS.logging_filename)
#     logging.info(f"Starting epoch: {start_epoch}, params count: {param_count(state.params)}")

#     return save_checkpoint_path
    

def main(argv):
    # Call preface to initialize variables
    preface()

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
    def energy_and_forces_vmap(graph: Graph):
        mod = hkflax.lift(message_passing, name=f'e3x_mlip')
        return mod(graph.features, graph.positions)

    opt = optax.adam(1e-3)

    # Restore or initialize the model state
    state, start_epoch = restore_or_initialize_state(
        key=init_key,
        model=energy_and_forces_vmap,
        opt=opt,
        data=samples_train,
        path=restore_checkpoint_dir,
        n_devices=n_devices
    )

    # Set up paths and logging for training process
    save_checkpoint_path = initialize_training(
        state=state,
        start_epoch=start_epoch,
        num_epochs=FLAGS.num_epochs,
        script_dir=os.path.dirname(os.path.abspath(__file__)),
        logging_filename=FLAGS.logging_filename,
        checkpoint_dir=FLAGS.checkpoint_dir
    )

    # Partial loss function with force weighting
    loss_fn_partial = partial(
        loss_fn_apply,
        model_apply=energy_and_forces_vmap.apply,
        forces_weight=FLAGS.forces_weight,
    )

    # Choose training method based on number of available devices
    if n_devices == 1:
        logging.info("Training without jax.pmap")
        run_scan(
            opt=opt,
            loss_fn_partial=loss_fn_partial,
            samples_train=samples_train,
            samples_valid=samples_valid,
            state=state,
            start_epoch=start_epoch,
            num_epochs=FLAGS.num_epochs,
            save_checkpoint_path=save_checkpoint_path,
            checkpointing_enabled=FLAGS.checkpointing_enabled,
            checkpoint_update_freq=FLAGS.checkpoint_update_freq,
            batch_size=FLAGS.batch_size,
            output_keys=FLAGS.output_keys,
            last_iter_info_only=False
        )
    
    elif n_devices > 1:
        logging.info("Training using jax.pmap")
        run_pmap(
            opt=opt,
            loss_fn_partial=loss_fn_partial,
            samples_train=samples_train,
            samples_valid=samples_valid,
            state=state,
            start_epoch=start_epoch,
            num_epochs=FLAGS.num_epochs,
            save_checkpoint_path=save_checkpoint_path,
            checkpointing_enabled=FLAGS.checkpointing_enabled,
            checkpoint_update_freq=FLAGS.checkpoint_update_freq,
            batch_size=FLAGS.batch_size,
            n_devices=n_devices,
            pmap_axis_name=pmap_axis_name,
            data_rng_key_generator=data_rng_key_generator,
            output_keys=FLAGS.output_keys,
        )

    print("Training complete. Well done!")

# Run the training process
if __name__ == '__main__':
    app.run(main)
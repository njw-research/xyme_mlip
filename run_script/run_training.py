# Import standard libraries
import os
import jax
import haiku as hk
from functools import partial
from absl import logging, app, flags
import haiku.experimental.flax as hkflax

from src.run.base import get_config
from src.utils.base import initialize_training
from src.utils.checkpoint.checkpoint_utils import find_highest_train_directory
from src.utils.checkpoint.checkpoint_state import restore_or_initialize_state
from src.utils.data.load_data import load_and_prepare_datasets
from src.utils.containers import Graph
from src.train.optimizer.optimizer import get_optimizer
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
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('gradient_clipping', 1000, 'Max norm graident.')

# Model hyperparameters
flags.DEFINE_integer('num_features', 8, 'Number of features')
flags.DEFINE_integer('max_degree', 2, 'Maximum degree')
flags.DEFINE_integer('num_iterations', 3, 'Number of iterations')
flags.DEFINE_integer('num_basis_functions', 16, 'Number of basis functions')
flags.DEFINE_float('cutoff', 5.0, 'Cutoff distance')
flags.DEFINE_integer('max_atomic_number', 9, 'Maximum atomic number')


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

def main(argv):

    config, n_devices = get_config(
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
        model_apply=energy_and_forces.apply,
        forces_weight=FLAGS.forces_weight,
    )

    # Choose training method based on number of available devices
    if n_devices == 1:
        logging.info("Training without jax.pmap")
        run_scan(
            opt=opt,
            loss_fn_partial=loss_fn_partial,
            samples_train=config['samples_train'],
            samples_valid=config['samples_valid'],
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
            samples_train=config['samples_train'],
            samples_valid=config['samples_valid'],
            state=state,
            start_epoch=start_epoch,
            num_epochs=FLAGS.num_epochs,
            save_checkpoint_path=save_checkpoint_path,
            checkpointing_enabled=FLAGS.checkpointing_enabled,
            checkpoint_update_freq=FLAGS.checkpoint_update_freq,
            batch_size=FLAGS.batch_size,
            n_devices=n_devices,
            pmap_axis_name=config['pmap_axis_name'],
            data_rng_key_generator=config['data_rng_key_generator'],
            output_keys=FLAGS.output_keys,
        )

    print("Training complete. Well done!")

# Run the training process
if __name__ == '__main__':
    app.run(main)
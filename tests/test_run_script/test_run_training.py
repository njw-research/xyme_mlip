# Import standard libraries
import os
from absl import logging, app, flags
import pytest
import shutil
# Import custom libraries
from src.run.base import RunInitializer
from src.train.scan.run_scan import run_scan
from src.train.pmap.run_pmap import run_pmap

# Define flags 
FLAGS = flags.FLAGS

# Define script directory
flags.DEFINE_string('script_dir', os.path.dirname(os.path.abspath(__file__)), "Script directory.")

# Set number of devics 
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

# Training paramaters 
flags.DEFINE_integer('num_epochs', 4, 'Number of training epochs.')
flags.DEFINE_integer('num_train', 4, 'Number of training samples.')
flags.DEFINE_integer('num_valid', 2, 'Number of validation samples.')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training.')
flags.DEFINE_float('forces_weight', 1.0, "Weight on forces loss.")
flags.DEFINE_bool('last_iter_info_only', False, "Only use last batch for info.")
flags.DEFINE_list('output_keys', ["grad_norm", "energy_mae", "forces_mae"], 'Output keys for logging.')

# Checkpointing and data
flags.DEFINE_string('training_output_dir', './train_/', 'Directory for saving all model outputs.')
flags.DEFINE_boolean('checkpointing_enabled', True, 'Enable checkpointing.')
flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'Directory for saving model checkpoints.')
flags.DEFINE_integer('checkpoint_update_freq', 2, 'Frequency of checkpoint updating')
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

def main(argv):
    # Initialize run
    initializer = RunInitializer(FLAGS)

    # Choose training method based on number of available devices
    if initializer.n_devices == 1:
        logging.info("Training without jax.pmap")
        run_scan(
            opt=initializer.opt,
            loss_fn_partial=initializer.loss_fn_partial,
            samples_train=initializer.config['samples_train'],
            samples_valid=initializer.config['samples_valid'],
            state=initializer.state,
            start_epoch=initializer.start_epoch,
            num_epochs=FLAGS.num_epochs,
            save_checkpoint_path=initializer.save_checkpoint_path,
            checkpointing_enabled=FLAGS.checkpointing_enabled,
            checkpoint_update_freq=FLAGS.checkpoint_update_freq,
            batch_size=FLAGS.batch_size,
            output_keys=FLAGS.output_keys,
            last_iter_info_only=FLAGS.last_iter_info_only
        )
    
    elif initializer.n_devices > 1:
        logging.info("Training using jax.pmap")
        run_pmap(
            opt=initializer.opt,
            loss_fn_partial=initializer.loss_fn_partial,
            samples_train=initializer.config['samples_train'],
            samples_valid=initializer.config['samples_valid'],
            state=initializer.state,
            start_epoch=initializer.start_epoch,
            num_epochs=FLAGS.num_epochs,
            save_checkpoint_path=initializer.save_checkpoint_path,
            checkpointing_enabled=FLAGS.checkpointing_enabled,
            checkpoint_update_freq=FLAGS.checkpoint_update_freq,
            batch_size=FLAGS.batch_size,
            n_devices=initializer.n_devices,
            pmap_axis_name=initializer.config['pmap_axis_name'],
            data_rng_key_generator=initializer.config['data_rng_key_generator'],
            output_keys=FLAGS.output_keys,
        )

    print("Training complete. Well done!")

def test_training_process():
    """Test that the training process finishes correctly."""
    try:
        # Run the main function as if it were called from the command line
        result = app.run(main)
        result = None
    except SystemExit as e:
        print(e.code)
        # Check if the system exits with a status code of 0, which indicates successful execution
        assert e.code == None

# Check that the output files exist and remove them
def test_files_exist():
    for output in ['train_0', 'train_samples.npz', 'valid_samples.npz']:  # List your output files here
        path = os.path.join(FLAGS.script_dir, output)
        assert os.path.exists(path)
        
        # Remove output files and directories after all tests are done
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


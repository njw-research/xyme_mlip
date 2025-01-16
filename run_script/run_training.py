# Import standard libraries
import os
from absl import logging, app, flags

# Import custom libraries
from src.train.scan.run_scan import run_scan
from src.train.pmap.run_pmap import run_pmap
from src.run.base import initalize_run

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

flags.DEFINE_string('script_dir', os.path.dirname(os.path.abspath(__file__)), "Script directory.")

# Set number of devics 
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

def main(argv):
    # Initialize run
    config, n_devices, state, start_epoch, save_checkpoint_path, opt, loss_fn_partial = initalize_run(FLAGS)

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
            last_iter_info_only=FLAGS.last_iter_info_only
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
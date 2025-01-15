import os 
import sys
import chex
from absl import logging
from jax.flatten_util import ravel_pytree
from src.utils.containers import TrainingState
from src.utils.setup_logging import setup_logging
from src.utils.checkpoint.checkpoint_utils import create_training_directory

def param_count(x: chex.ArrayTree) -> int:
    """Count the number of parameters in a PyTree of parameters."""
    return ravel_pytree(x)[0].shape[0]

# Initialize training process and directories
def initialize_training(state: TrainingState, 
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
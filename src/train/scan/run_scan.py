from absl import logging
from functools import partial
from src.utils.containers import TrainingState
from src.train.scan.training_scan import train_scan, create_scan_epoch_fn, create_validation_epoch_fn
from src.train.steps.steps import training_step, validation_step


def run_scan(
    opt,
    loss_fn_partial,
    samples_train,
    samples_valid,
    state,
    start_epoch,
    num_epochs,
    save_checkpoint_path,
    checkpointing_enabled,
    checkpoint_update_freq,
    batch_size,
    output_keys,
    last_iter_info_only=False
):
    """
    Run the training and validation process for a model.

    Args:
        opt: The optimizer to use for training.
        general_ml_loss_fn_partial: The loss function to use for training and validation.
        samples_train: Training data samples.
        samples_valid: Validation data samples.
        state: The initial state containing model parameters and optimizer state.
        start_epoch: The starting epoch for training.
        num_epochs: The total number of epochs to train.
        save_checkpoint_path: The directory path to save model checkpoints.
        checkpointing_enabled: Boolean flag to enable or disable checkpointing.
        checkpoint_update_freq: Frequency of checkpoint updates.
        batch_size: The batch size for training and validation.
    """
    
    # Define the training step function with the necessary parameters
    training_step_fn = partial(training_step, optimizer=opt, loss_fn=loss_fn_partial, use_pmap=False)

    # Create the training epoch function using scan
    scan_epoch_fn = create_scan_epoch_fn(
        training_step_fn,
        data=samples_train,
        last_iter_info_only=last_iter_info_only,
        batch_size=batch_size
    )

    def update_fn(state):
        """
        Update the training state after each epoch.

        Args:
            state: The current training state containing parameters and optimizer state.

        Returns:
            tuple: Updated training state and additional information.
        """
        # Run the scan epoch function and update the training state
        params, opt_state, key, info = scan_epoch_fn(state.params, state.opt_state, state.key)
        return TrainingState(params, opt_state, key), info

    # Define the validation step function with the necessary parameters
    valid_step_fn = partial(validation_step, loss_fn=loss_fn_partial)

    # Create the validation epoch function using scan
    scan_epoch_valid_fn = create_validation_epoch_fn(
        valid_step_fn,
        data=samples_valid,
        last_iter_info_only=last_iter_info_only,
        batch_size=batch_size
    )

    def valid_fn(state):
        """
        Validate the model after each epoch.

        Args:
            state: The current training state containing model parameters.

        Returns:
            info: Validation information.
        """
        # Run the scan epoch validation function
        info = scan_epoch_valid_fn(state.params, state.key)
        return info

    # Run the training process with the defined update and validation functions
    state = train_scan(
        state=state, 
        start_epoch=start_epoch, 
        num_epochs=num_epochs, 
        update_fn=update_fn, 
        valid_fn=valid_fn, 
        checkpointing_enabled=checkpointing_enabled, 
        checkpoint_dir=save_checkpoint_path,
        checkpoint_update_freq=checkpoint_update_freq,
        output_keys=output_keys
    )
    
    logging.info("Training completed. Well done!")

    return state

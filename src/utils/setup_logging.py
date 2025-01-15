import os
from absl import flags, logging

# Define FLAGS object globally for consistency across the script
FLAGS = flags.FLAGS

def setup_logging(output_dir, logging_filename):
    """
    Sets up logging for the application using the Abseil library.
    
    Args:
        output_dir (str): Directory where the log file will be created.
        logging_filename (str): Name of the log file to write log messages to.
    
    This function initializes logging to write log messages to a specified
    file and sets the logging verbosity to INFO. It also logs all current
    flag values for debugging purposes.
    """
    log_file_path = os.path.join(output_dir, logging_filename)
    logging.get_absl_handler().use_absl_log_file(log_file_path)
    logging.set_verbosity(logging.INFO)
    logging.info("Logging all flags:")
    for flag_name in sorted(FLAGS):
        flag_value = FLAGS[flag_name].value
        logging.info("%s: %s", flag_name, flag_value)
import os
import re
import glob

def find_highest_train_directory(base_dir):
    """
    Finds the training directory with the highest numerical suffix.
    
    Args:
        base_dir (str): Base directory where training directories are located.
    
    Returns:
        str: Path to the training directory with the highest index,
             or None if no training directory exists.
    
    This function scans the specified base directory and identifies the
    training directory with the highest numerical suffix (e.g., train_0, train_1, etc.).
    """
    pattern = re.compile(r'train_(\d+)')
    highest_index = -1
    for entry in os.listdir(base_dir):
        match = pattern.match(entry)
        if match:
            index = int(match.group(1))
            if index > highest_index:
                highest_index = index
    return os.path.join(base_dir, f'train_{highest_index}') if highest_index >= 0 else None

def create_training_directory(base_dir):
    """
    Creates a new training directory with an incremented number.
    
    Args:
        base_dir (str): Base directory where the training directories will be created.
    
    Returns:
        str: Path to the newly created training directory.
    
    This function checks for existing training directories and creates a new one
    by incrementing the numerical suffix until a unique name is found.
    """
    index = 0
    while True:
        new_dir = os.path.join(base_dir, f'train_{index}')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        index += 1

def find_latest_checkpoint(directory):
    """
    Finds the latest checkpoint file in the specified directory.
    
    Args:
        directory (str): Directory to search for checkpoint files.
    
    Returns:
        str or None: Path to the latest checkpoint file, or None if no files are found.
    
    This function retrieves all checkpoint files with a .pkl extension and
    returns the one with the most recent modification time.
    """
    checkpoints = glob.glob(os.path.join(directory, '*.pkl'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

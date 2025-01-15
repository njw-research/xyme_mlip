import os
from absl import logging
import urllib.request

def download_dataset(data_path, filename):
    """Downloads the dataset if not already available."""
    if not os.path.exists(data_path):
        logging.info(f"Downloading {filename} (this may take a while)...")
        urllib.request.urlretrieve(f"http://www.quantum-machine.org/gdml/data/npz/{filename}", data_path)
        logging.info(f"Download complete: {filename}")
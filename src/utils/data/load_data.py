import os
import numpy as np
import jax.numpy as jnp
from src.utils.containers import Graph
from src.utils.data.download_data_from_gdml import download_dataset
from src.utils.data.prepare_datasets_from_gdml import prepare_datasets


def load_and_prepare_datasets(script_dir, data_key, data_dir, dataset, num_train, num_valid):
    """
    Load and prepare the training and validation datasets.
    
    Args:
        script_dir (str): Directory where scripts and temporary files are stored.
        data_key (str): Key used to identify the dataset.
        data_dir (str): Directory where the dataset is stored.
        dataset (str): Name of the dataset file to load.
        num_train (int): Number of training samples to load.
        num_valid (int): Number of validation samples to load.
    
    Returns:
        tuple: A tuple containing:
            - samples_train (FullGraphSample): Training dataset sample.
            - samples_valid (FullGraphSample): Validation dataset sample.
            - atomic_numbers (ndarray): Array of atomic numbers used in the dataset.
            - n_nodes (int): Number of nodes in the dataset.
    
    This function attempts to load existing training and validation samples from
    temporary files. If those files do not exist, it downloads the dataset,
    prepares it, and saves the prepared samples for future use. The samples are
    centered around the mean position.
    """
    data_path = os.path.join(data_dir, dataset)
    download_dataset(data_path, dataset)
    
    train_samples_file = os.path.join(script_dir, 'train_samples.npz')
    valid_samples_file = os.path.join(script_dir, 'valid_samples.npz')

    if os.path.exists(train_samples_file) and os.path.exists(valid_samples_file):
        print("Loading dataset")
        train_data = np.load(train_samples_file)
        features = train_data['features'][0]
        n_nodes = len(features)

        samples_train = Graph(positions=train_data['positions'], features=train_data['features'], energy=train_data['energy'], forces=train_data['forces'])

        valid_data = np.load(valid_samples_file)
        samples_valid = Graph(positions=valid_data['positions'], features=valid_data['features'], energy=valid_data['energy'], forces=valid_data['forces'])
    else:
        print("Downloading dataset")
        dataset_loaded = np.load(data_path)
        train_data, valid_data, features, n_nodes, mean_energy = prepare_datasets(data_key, dataset_loaded, num_train=num_train, num_valid=num_valid)

        samples_train = Graph(positions=train_data['positions'], features=jnp.tile(features[:, None], (num_train, 1, 1)), energy=train_data['energy'], forces=train_data['forces'])
        samples_valid = Graph(positions=valid_data['positions'], features=jnp.tile(features[:, None], (num_valid, 1, 1)), energy=valid_data['energy'], forces=valid_data['forces'])

        # Centering the positions of training samples around the center of mass
        centre_of_mass_x = jnp.mean(samples_train.positions, axis=-2, keepdims=True)
        samples_train = samples_train._replace(positions=samples_train.positions - centre_of_mass_x)

        # Centering the positions of validation samples around the center of mass
        centre_of_mass_x = jnp.mean(samples_valid.positions, axis=-2, keepdims=True)
        samples_valid = samples_valid._replace(positions=samples_valid.positions - centre_of_mass_x)

        # Save the prepared samples for future use
        np.savez(train_samples_file, positions=np.array(samples_train.positions), features=np.array(samples_train.features), energy=np.array(samples_train.energy), forces=np.array(samples_train.forces))
        np.savez(valid_samples_file, positions=np.array(samples_valid.positions), features=np.array(samples_valid.features), energy=np.array(samples_valid.energy), forces=np.array(samples_valid.forces))

    return samples_train, samples_valid, features, n_nodes
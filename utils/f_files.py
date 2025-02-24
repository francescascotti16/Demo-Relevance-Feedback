
import pandas as pd
import h5py
import tqdm
from utils.f_process_data import *
from utils.functions_similarity_metrics import logistic
import requests
import os 

def create_logistic_indexed_data(indexed_data):
    """
    Create a logistic transformation of the indexed data.
    parameters:
    - indexed_data (numpy.ndarray): Array containing indexed data.
    Returns:
    - np.ndarray: Logistic transformation of the indexed data.
    This function applies a logistic transformation to the indexed data.
    """
    indexed_data_logistic = np.copy(indexed_data)

    for i in range(indexed_data.shape[0]):
        indexed_data_logistic[i] = logistic(indexed_data[i])  # Assicurati che logistic sia una funzione valida

    return indexed_data_logistic


def read_indexed_hdf5_and_create_index(hdf5_indexed_file_path):
    """
    Read an indexed HDF5 file and create an index dictionary mapping ids to indices.

    Parameters:
    - hdf5_indexed_file_path (str): Path to the indexed HDF5 file.

    Returns:
    - dict: Dictionary mapping ids to their corresponding indices.
    - np.ndarray: Indexed data.
    - np.ndarray: Indexed ids.

    This function reads 'data' and 'ids' from the HDF5 file,
    and creates a dictionary ('index_2_id') mapping ids to indices.
    """
    index_2_id = {}

    with h5py.File(hdf5_indexed_file_path, 'r') as hdf5_file:
        # Read the 'data' and 'ids' groups
        indexed_data = hdf5_file['indexed_data']['data'][:]
        indexed_ids = hdf5_file['indexed_data']['ids'][:]
        
        # Create index_2_id dictionary
        for i, id in enumerate(indexed_ids):
            id_str = id.decode('utf-8') if isinstance(id, bytes) else id
            index_2_id[id_str] = i

    return index_2_id, indexed_data, indexed_ids  

def create_indexed_hdf5(original_file_path, new_file_path, batch_size=1000):
    """
    Create an indexed HDF5 file from the original HDF5 file.

    Parameters:
    - original_file_path (str): Path to the original HDF5 file.
    - new_file_path (str): Path to the new indexed HDF5 file to be created.
    - batch_size (int, optional): Size of batches to process data entries. Default is 1000.

    Returns:
    - str: Path of the newly created indexed HDF5 file.

    This function reads data and ids from the original HDF5 file, and stores them
    in batches into a new HDF5 file. If the new file already exists, it will be deleted
    before creating the new indexed file.
    """
    # Delete the existing file if it exists
    if os.path.exists(new_file_path):
        os.remove(new_file_path)

    # Open the original HDF5 file
    with h5py.File(original_file_path, 'r') as original_file:
        # Read the 'data' and 'ids' groups
        data = original_file['data']
        ids = original_file['ids']
        ids = ids[:]  # Read all ids into memory (assuming they are strings)

        # Calculate total number of data entries
        total_entries = len(ids)

        # Create a new HDF5 file to store the indexed data
        with h5py.File(new_file_path, 'w') as new_file:
            # Create a group for the indexed data
            indexed_group = new_file.create_group('indexed_data')
            
            # Create datasets for ids and data
            id_dataset = indexed_group.create_dataset('ids', (total_entries,), maxshape=(None,), dtype=ids.dtype)
            data_dataset = indexed_group.create_dataset('data', (total_entries,) + data.shape[1:], maxshape=(None,) + data.shape[1:], dtype=data.dtype)

            # Store data entries in batches with a progress bar
            for i in tqdm(range(0, total_entries, batch_size), desc="Processing Batches"):
                batch_ids = ids[i:i+batch_size]
                batch_data = data[i:i+batch_size]

                # Write the batch to the datasets
                id_dataset[i:i+batch_size] = batch_ids
                data_dataset[i:i+batch_size] = batch_data

    return new_file_path  # Return the path of the newly created indexed HDF5 file


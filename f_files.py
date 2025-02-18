
import pandas as pd
import h5py
import tqdm
from f_process_data import *
from functions_similarity_metrics import logistic
import requests
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


    for i in tqdm(range(indexed_data.shape[0]), desc="Processing"):
        indexed_data_logistic[i] = logistic(indexed_data[i])

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

    This function reads 'data' and 'ids' from the 'indexed_data' group of the HDF5 file,
    and creates a dictionary ('index_2_id') mapping ids to indices.
    """
    index_2_id = {}

    with h5py.File(hdf5_indexed_file_path, 'r') as hdf5_file:
        # Read the 'data' and 'ids' groups
        indexed_data = hdf5_file['indexed_data/data'][:]
        indexed_ids = hdf5_file['indexed_data/ids'][:]
        
        # Create index_2_id dictionary
        for i, id in enumerate(indexed_ids):
            id_str = id.decode('utf-8') if isinstance(id, bytes) else id
            index_2_id[id_str] = i

    return index_2_id, indexed_data ,indexed_ids # Return the index dictionary


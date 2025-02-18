
import pandas as pd
import h5py
import tqdm
#from scipy.special import logistic
from f_process_data import *




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

# Applica la funzione logistica ad ogni vettore (riga) individualmente con una barra di progresso
    for i in tqdm(range(indexed_data.shape[0]), desc="Processing"):
        indexed_data_logistic[i] = logistic(indexed_data[i])

    return indexed_data_logistic

import requests

def fetch_text_feature(text):
    # Construct the URL by encoding the text string
    text = text.lower()
    url = f"https://visione.isti.cnr.it/services/features-clip-laion/get-text-feature?text={text.replace(' ', '+')}&normalized=true"
    
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the response is successful
        response.raise_for_status()  # Raises HTTPError for bad responses
        
        # Parse the JSON response
        data = response.json()

        # Store the data in an array
        result_array = [data]
        
        # Output the result array
        
        return result_array
    
    except requests.exceptions.RequestException as e:
        # Handle any error that occurs during the request
        print(f"Error: {e}")

# Example usage
array=fetch_text_feature('a man with a dog')

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



def create_dataframe_from_results(df_results, index_2_id, indexed_data, indexed_ids):
    """
    Create a DataFrame from results CSV file, using index mapping and indexed data.

    Parameters:
    - results (pd.DataFrame): DataFrame containing results from the search engine.
    - index_2_id (dict): Dictionary mapping ids to their corresponding indices.
    - indexed_data (numpy.ndarray): Array containing indexed data.
    - indexed_ids (numpy.ndarray): Array containing indexed ids.
    

    Returns:
    - pd.DataFrame: DataFrame with columns named by ids_sorted and filled with features.

    This function reads ids from the first column of the results CSV file,
    retrieves their indices using index_2_id mapping, and creates a DataFrame
    with columns named by ids_sorted and filled with corresponding features.
    """
    


    ids_to_retrieve = df_results['imgId'].tolist()
   
    # Retrieve indices and sort them
    indices_to_retrieve = [index_2_id.get(id) for id in ids_to_retrieve]
    indices_to_retrieve = [idx for idx in indices_to_retrieve if idx is not None]  # Filter out None values
    indices_to_retrieve.sort()
    
    # Sort ids according to retrieved indices
    ids_sorted = [indexed_ids[i].decode('utf-8') for i in indices_to_retrieve]
    
    # Create DataFrame with columns named by ids_sorted
    df = pd.DataFrame(columns=ids_sorted)
    
    # Populate DataFrame with corresponding features
    features = indexed_data[indices_to_retrieve]
    
    for i, col in enumerate(df.columns):
        df[col] = pd.Series(features[i])
   
    return df  # Return the resulting DataFrame


import requests

import pandas as pd


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

import pandas as pd
import numpy as np

def create_dataframe_from_results(df_results, index_2_id, indexed_data, indexed_ids, shuffle=True, seed=42):
    """
    Create a DataFrame from results CSV file, using index mapping and indexed data.

    Parameters:
    - df_results (pd.DataFrame): DataFrame containing results from the search engine.
    - index_2_id (dict): Dictionary mapping ids to their corresponding indices.
    - indexed_data (numpy.ndarray): Array containing indexed data.
    - indexed_ids (numpy.ndarray): Array containing indexed ids.
    - shuffle (bool, optional): Whether to shuffle the columns. Default is True.
    - seed (int, optional): Random seed for shuffling the columns. Default is 42.

    Returns:
    - pd.DataFrame: DataFrame with or without shuffled columns, based on shuffle parameter.
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

    # Shuffle the columns if shuffle=True
    if shuffle:
        np.random.seed(seed)
        shuffled_columns = np.random.permutation(df.columns)
        df = df[shuffled_columns]

    return df  # Return the DataFrame


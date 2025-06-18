import requests
import pandas as pd
import numpy as np

def compute_first_probability(data_df, initial_query_emb, tau=11.6):
    """
    Compute the first probability distribution based on the initial query embedding.
    
    Parameters:
    - data_df (pd.DataFrame): DataFrame with embedding vectors (rows = features).
    - initial_query_emb (np.ndarray): Initial query embedding vector.
    - tau (float): Temperature parameter for softmin. Default is 11.6.
    
    Returns:
    - np.ndarray: Probability distribution as a 1D array.
    """
    initial_query_emb = initial_query_emb.reshape(-1, 1)  # Ensure column vector
    arr = data_df.values.astype(np.float64)              # Convert to numpy array
    diff = arr - initial_query_emb                       # Element-wise difference
    dist = np.linalg.norm(diff, axis=0)                  # Euclidean distances

    num = np.exp(-dist / tau)                            # Softmin numerator
    den = np.sum(num)                                    # Softmin denominator

    if not np.isfinite(den) or den == 0:                 # Check for valid denominator
        return np.zeros_like(num)

    return num / den                                     # Return probability distribution


def fetch_text_feature(text):
    # Construct the URL by encoding the text string
    text = text.lower()
    url = f"https://visione.isti.cnr.it/services/features-clip-laion/get-text-feature?text={text.replace(' ', '+')}&normalized=true"
    
    try:
        # Send a GET request to the URL
        response = requests.get(url,verify=False)
        
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

def create_dataframe_from_results(df_results, index_2_id, indexed_data, indexed_ids):
    """
    Create a DataFrame from results CSV file, using index mapping and indexed data.

    Parameters:
    - df_results (pd.DataFrame): DataFrame containing results from the search engine.
    - index_2_id (dict): Dictionary mapping ids to their corresponding indices.
    - indexed_data (numpy.ndarray): Array containing indexed data.
    - indexed_ids (numpy.ndarray): Array containing indexed ids.

    Returns:
    - pd.DataFrame: DataFrame with features in the same column order as in df_results['imgId'].
    """
    ids_to_retrieve = df_results['imgId'].tolist()

    ids_sorted = []
    features = []

    for img_id in ids_to_retrieve:
        idx = index_2_id.get(img_id)
        if idx is not None:
            ids_sorted.append(indexed_ids[idx].decode('utf-8'))
            features.append(indexed_data[idx])

    # Transpose to get features as columns
    df = pd.DataFrame(data=np.array(features).T, columns=ids_sorted)

    return df

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from datetime import datetime
import hnswlib  # HNSW index library

# HNSW Index Initialization
def build_hnsw_index(data_df, space='l2', ef_construction=200, M=16):
    """
    Builds an HNSW index for fast approximate nearest neighbor searches.

    Parameters:
    data_df: DataFrame with the dataset (each column is a vectorized image).
    space: Metric space, default is 'l2'.
    ef_construction: Construction time/accuracy trade-off parameter.
    M: Number of bi-directional links for each element.

    Returns:
    index: The constructed HNSW index.
    """
    dim = data_df.shape[0]  # Dimensionality of the vectors
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=data_df.shape[1], ef_construction=ef_construction, M=M)
    
    # Add data to the index
    data = data_df.values.T.astype(np.float32)  # HNSW expects float32
    index.add_items(data, ids=np.arange(data_df.shape[1]))

    # Set runtime parameters
    index.set_ef(50)  # Trade-off between accuracy and speed
    return index

# Modified MSED computation function with HNSW integration
def get_msed_with_hnsw(data_df, entropy_dict, relevant_ids, non_relevant_ids, old_score, alpha, beta, gamma, precomputed_dict, index):
    """Calculate MSED scores using the HNSW index for efficiency."""
    start_time = datetime.now()
    msed_score = []

    precomputed_sum_neg = precomputed_dict['precomputed_sum_neg']
    precomputed_sum_pos = precomputed_dict['precomputed_sum_pos']
    precomputed_sum_entropy_neg = precomputed_dict['precomputed_sum_entropy_neg']
    precomputed_sum_entropy_pos = precomputed_dict['precomputed_sum_entropy_pos']
    n_pos = precomputed_dict['n_pos']
    n_neg = precomputed_dict['n_neg']

    # Update precomputed sums for relevant and non-relevant IDs
    for el in relevant_ids:
        numpy_data = data_df[el].to_numpy().reshape(1, -1)
        precomputed_sum_pos += numpy_data
        precomputed_sum_entropy_pos += entropy_dict[el]
        n_pos += 1

    for el in non_relevant_ids:
        numpy_data = data_df[el].to_numpy().reshape(1, -1)
        precomputed_sum_neg += numpy_data
        precomputed_sum_entropy_neg += entropy_dict[el]
        n_neg += 1

    new_dict = {
        'precomputed_sum_pos': precomputed_sum_pos,
        'precomputed_sum_neg': precomputed_sum_neg,
        'precomputed_sum_entropy_neg': precomputed_sum_entropy_neg,
        'precomputed_sum_entropy_pos': precomputed_sum_entropy_pos,
        'n_pos': n_pos,
        'n_neg': n_neg
    }

    # Iterate over data using the HNSW index for nearest neighbors
    for img_id in range(data_df.shape[1]):
        if img_id in relevant_ids or img_id in non_relevant_ids:
            continue

        data_vec = data_df.iloc[:, img_id].to_numpy().reshape(1, -1)
        data_entropy = entropy_dict[img_id]

        mean_pos = (precomputed_sum_pos + data_vec) / (n_pos + 1)
        entropy_numerator_pos = shannon_entropy(mean_pos, axis=1)[0]
        avg_entropy_denominator_pos = (precomputed_sum_entropy_pos + data_entropy) / (n_pos + 1)
        score_pos = n_pos + 1 - np.exp(entropy_numerator_pos - avg_entropy_denominator_pos)

        mean_neg = (precomputed_sum_neg + data_vec) / (n_neg + 1)
        entropy_numerator_neg = shannon_entropy(mean_neg, axis=1)[0]
        avg_entropy_denominator_neg = (precomputed_sum_entropy_neg + data_entropy) / (n_neg + 1)
        score_neg = n_neg + 1 - np.exp(entropy_numerator_neg - avg_entropy_denominator_neg)

        score = beta * score_pos - gamma * score_neg
        msed_score.append(score)

    new_scores = alpha * old_score + np.array(msed_score).flatten()
    end_time = datetime.now()
    total_time = end_time - start_time
    return new_scores, new_dict, total_time

# Integration Example
if __name__ == "__main__":
    # Example dataset
    data = np.random.random((384, 1000))  # 1000 images with 384-dimensional vectors
    data_df = pd.DataFrame(data)

    # Example entropy dictionary
    entropy_dict = {i: np.random.random() for i in range(data_df.shape[1])}

    # Relevant and non-relevant image IDs (example)
    relevant_ids = [0, 1, 2]
    non_relevant_ids = [3, 4]

    # Precomputed dictionary initialization
    precomputed_dict = {
        'precomputed_sum_pos': np.zeros((1, 384)),
        'precomputed_sum_neg': np.zeros((1, 384)),
        'precomputed_sum_entropy_neg': 0,
        'precomputed_sum_entropy_pos': 0,
        'n_pos': 0,
        'n_neg': 0
    }

    # Build HNSW index
    hnsw_index = build_hnsw_index(data_df)

    # Compute MSED with HNSW
    old_score = np.zeros(data_df.shape[1])
    alpha, beta, gamma = 1.0, 0.7, 0.7
    new_scores, updated_dict, computation_time = get_msed_with_hnsw(
        data_df, entropy_dict, relevant_ids, non_relevant_ids, old_score, alpha, beta, gamma, precomputed_dict, hnsw_index
    )

    print("New scores:", new_scores)
    print("Computation time:", computation_time)

import numpy as np
import pandas as pd
from utils.f_display_and_feedback import create_display
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime
def distance_from_hyperplane(w, b, data_df):
    '''
    Computes the distance of each data point from the hyperplane defined by the SVM.

    Parameters:
    - w (np.array): The weight vector of the trained SVM classifier.
    - b (float): The bias term of the trained SVM classifier.
    - data_df (pd.DataFrame): A DataFrame where columns represent feature vectors of the dataset.

    Returns:
    - distance (np.array): An array containing the computed distances for each data point.
    - total_time (timedelta): The time taken to compute the distances.
    '''
    start_time = datetime.now()

    # Compute distance from the hyperplane for each data point
    distance = np.dot(data_df.T, w) + b

    end_time = datetime.now()
    total_time = end_time - start_time

    return distance, total_time


def svm_score(data_df, relevant, non_relevant):
    '''
    Trains a linear SVM classifier using the provided relevant and non-relevant examples 
    and computes the distances of all dataset points from the resulting decision boundary.

    Parameters:
    - data_df (pd.DataFrame): A DataFrame where columns represent feature vectors of the dataset.
    - relevant (np.array): A 2D NumPy array containing the feature vectors of relevant items.
    - non_relevant (np.array): A 2D NumPy array containing the feature vectors of non-relevant items.

    Returns:
    - distances_to_hyper (np.array): An array containing the computed distances for each data point.
    - time_distance_from_hyperplane (timedelta): The time taken to compute distances from the hyperplane.
    - total_time (timedelta): The total execution time of the function.
    '''
    start_time = datetime.now()

    # Create training data by combining relevant and non-relevant feature vectors
    X = np.concatenate((relevant, non_relevant), axis=0)

    # Create labels: 1 for relevant, 0 for non-relevant
    y = np.concatenate((np.ones(relevant.shape[0]), np.zeros(non_relevant.shape[0])))

    # Train a linear SVM classifier
    clf = SVC(kernel='linear')
    clf.fit(X, y)

    # Retrieve the weight vector and bias term of the trained classifier
    w = clf.coef_[0]
    b = clf.intercept_[0]

    # Compute classification accuracy (for reference, not used in output)
    accuracy = accuracy_score(y, clf.predict(X))

    # Compute distances from the hyperplane for all data points
    distances_to_hyper, time_distance_from_hyperplane = distance_from_hyperplane(w, b, data_df)

    end_time = datetime.now()
    total_time = end_time - start_time

    return distances_to_hyper, time_distance_from_hyperplane, total_time

def svm_single_step(data_df, display_df, relevant_ids, non_relevant_ids, initial_scores=None, alpha=0.75, beta=0.25):
    '''
    Performs a single step of the SVM-based relevance feedback algorithm.

    Parameters:
    - data_df (pd.DataFrame): A DataFrame where columns represent feature vectors of the dataset.
    - display_df (pd.DataFrame): A DataFrame containing the current set of displayed items.
    - relevant_ids (list): List of indices corresponding to relevant items in data_df.
    - non_relevant_ids (list): List of indices corresponding to non-relevant items in data_df.
    - initial_scores (np.array, optional): An array of previous scores for each item. Defaults to None.
    - alpha (float, optional): Weight for previous scores in score updating. Defaults to 0.
    - beta (float, optional): Weight for new scores in score updating. Defaults to 1.

    Returns:
    - display_df (pd.DataFrame): Updated DataFrame of items to be displayed.
    - new_scores (np.array): Updated scores for each item in the dataset.
    - total_time_single_step (timedelta): Total execution time for the function.
    - time_distance_from_hyperplane (timedelta): Time taken to compute distances from the hyperplane.
    - total_time_score_computation (timedelta): Time taken to compute the new scores.
    - total_time_display (timedelta): Time taken to update the display.
    '''
    start_time = datetime.now()
    n_display = display_df.shape[1]

    old_scores = initial_scores
    # Initialize the query with zero if not provided
    if old_scores is None:
        old_scores = np.array([0] * data_df.shape[1])

    # If there are no relevant items, return the current display and scores
    if len(relevant_ids) == 0:
        return display_df, old_scores

    # Extract relevant and non-relevant feature vectors
    relevant = data_df[relevant_ids].to_numpy().T
    non_relevant = data_df[non_relevant_ids].to_numpy().T

    # Compute new scores using SVM
    new_scores_1, time_distance_from_hyperplane, total_time_score_computation = svm_score(
        data_df, relevant, non_relevant
    )

    # Update scores using weighted combination of old and new scores
    new_scores = alpha * old_scores + beta * new_scores_1

    # Update the display based on new scores
    display_df, total_time_display = create_display(
        data_df, new_scores, n_display, is_ascending=False
    )

    end_time = datetime.now()
    total_time_single_step = end_time - start_time

    return display_df, new_scores, total_time_single_step, time_distance_from_hyperplane, total_time_score_computation, total_time_display

import pandas as pd
import numpy as np

import random as random

def get_actions(display_df_t, groundtruth_dic, already_selected_images, k_pos=-1, k_neg=-1, no_reclick_image=True):
    '''
    Parameters:
    display_df_t: DataFrame with the display one column for each image  
    groundtruth_dic: dictionary with the groundtruth relevance of the images
    already_selected_images: list of images already selected (cannot be selected again))
    k_pos: number of positive images to select
    k_neg: number of negative images to select
    no_reclick_image: boolean to avoid reclicking on the same image
    
    Returns:
    actions: list of tuples with the selected images and their relevance
    list_display_relevance: list of the relevance of the images in the display
    '''
    random.seed(42)
    # Function to get the action vector from a display
    positive_action = []
    negatives_action = []
    list_display_relevance=[] #store the relevance of the images in the display
    for image in display_df_t.columns:
        list_display_relevance.append(groundtruth_dic[image])
        if no_reclick_image:
            if (groundtruth_dic[image] == 1) and (image not in already_selected_images):
                positive_action.append(image)
            if (groundtruth_dic[image] == 0) and (image not in already_selected_images):
                negatives_action.append(image)
        else:
            if (groundtruth_dic[image] == 1): #not used in teh experiments but might be useful to test in the incremental case
                positive_action.append(image)
            if (groundtruth_dic[image] == 0):
                negatives_action.append(image)



    #if k_pos !=-1 and k_pos is smaller than the number of positive images in the display we randomly select k_pos positive images 
    if k_pos !=-1  and k_pos < len(positive_action):
        np.random.shuffle(positive_action) #random shaffle the positive images and select the first k_pos
        positive_action = positive_action[:k_pos]

    #if k_neg !=-1 and k_neg is smaller than the number of negative images in the display we randomly select k_neg negative images
    if k_neg != -1 and k_neg < len(negatives_action):
        np.random.shuffle(negatives_action) #random shaffle the negative images and select the first k_neg
        negatives_action = negatives_action[:k_neg]
    
    positive_actions=[(action, 1) for action in positive_action]
    negative_actions=[(action, 0) for action in negatives_action]
    actions = positive_actions + negative_actions
    return actions, list_display_relevance 

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def order_first_display(df, n_display, initial_query, similarity="euclidean"):
    """
    Orders the columns of the DataFrame based on their distance (Euclidean or cosine)
    from the initial query embedding, and returns the top n_display columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with image vectors as columns.
    n_display : int
        Number of images to select.
    initial_query : np.ndarray
        Initial query vector, shape (n_features,) or (n_features, 1).
    similarity : str
        Type of distance to use: "euclidean" or "cosine".
    """
    # Removes the last row of the DataFrame
    df = df.iloc[:-1]

    # Ensures that initial_query has shape (n_features, 1)
    query_vec = initial_query.reshape(-1, 1)

    if similarity == "cosine":
        # Compute cosine similarity and sort in descending order (more similar → higher)
        sims = cosine_similarity(df.values.T, query_vec.T).flatten()
        sorted_idx = np.argsort(-sims)  # less similar at the end
    else:
        # Compute classic Euclidean distance
        dists = np.linalg.norm(df.values - query_vec, axis=0)
        sorted_idx = np.argsort(dists)

    # Select the top n_display indices
    first_sorted_ids = sorted_idx[:n_display]

    # Get the corresponding columns
    sorted_columns = df.columns[first_sorted_ids]

    # Return only the selected columns
    first_display = df.loc[:, sorted_columns]
    
    return first_display


def sample_first_display(df, seed,n_display, indexes_positive_initial_images):
    '''
    Parameters:
    df: DataFrame with the dataset one column for each image and the last row with the GT relevance
    seed: int,  seed for the random sampling
    n_display: int,  number of images in the display
    indexes_positive_initial_images: list of indexes of the relevant images in the first display.
    
    Returns:
    first_display: DataFrame with the first display
    '''
    # Function to sample and transpose the display dataset
    #np.random.seed(seed)  # select the seed for numpy
    selected_columns = df.columns[df.iloc[-1, :] == 1] 
    #filtered_df = df.loc[:, selected_columns]
    #positive_display_df=filtered_df[filtered_df.iloc[:, indexes_positive_initial_images]]  # select  relevant images to be shown in the display (using indexes_positive_initial_pairs)
    selected_columns=selected_columns[indexes_positive_initial_images]
    positive_display_df = df.loc[:, selected_columns]
    n_positive=len(indexes_positive_initial_images)
    n_negative=max(n_display-n_positive,0)

    non_relevant_df = df.loc[:, df.iloc[-1] == 0] #non_relevant images   
    negative_display_df = non_relevant_df.sample(n=n_negative, axis=1,  random_state=seed) # select n_display-2 non_relevant images to be shown in the display
    first_display = pd.concat([positive_display_df, negative_display_df], axis=1).iloc[:-1, :]
    return first_display

def create_display(data_df, score, n_display, is_ascending=False):
    
    '''
    Parameters:
    data_df: DataFrame with the dataset one column for each image and the last row with the GT relevance
    score: list with the score of each image
    n_display: int, number of images in the display
    is_ascending: boolean, to sort the display in ascending order
    
    Returns:
    display_df: DataFrame with the display
    '''
   
    # Function to create and transpose the display dataset
    df_copy_t = data_df.copy().transpose()  
    if score.ndim > 1:
        score = np.linalg.norm(score, axis=1)

    df_copy_t['score'] = score
    
    df_sorted = df_copy_t.sort_values(by='score', ascending=is_ascending)
    display_df_t = df_sorted.head(n_display)
    display_df_t = display_df_t.copy()
    display_df_t.drop(columns=['score'], errors='ignore', inplace=True)
    display_df = display_df_t.transpose()
    
  
    return display_df

def create_display_svm(data_df, score, n_display, is_ascending=False):
    
    '''
    Parameters:
    data_df: DataFrame with the dataset one column for each image and the last row with the GT relevance
    score: list with the score of each image
    n_display: int, number of images in the display
    is_ascending: boolean, to sort the display in ascending order
    
    Returns:
    display_df: DataFrame with the display
    '''
   
    # Function to create and transpose the display dataset
    df_copy_t = data_df.copy().transpose()  
    

    df_copy_t['score'] = score
    
    df_sorted = df_copy_t.sort_values(by='score', ascending=is_ascending)
    display_df_t = df_sorted.head(n_display)
    display_df_t = display_df_t.copy()
    display_df_t.drop(columns=['score'], errors='ignore', inplace=True)
    display_df = display_df_t.transpose()
    
  
    return display_df
# def get_actions(display_df_t, groundtruth_dic, already_selected_images, k_pos=-1, k_neg=-1, no_reclick_image=True):
#     # Initialize lists to store positive and negative actions, and the relevance values for displayed images
#     positive_action = []
#     negatives_action = []
#     list_display_relevance = []
    
#     # Iterate through each image in the current display set
#     for image in display_df_t.columns:
#         # Append the ground truth relevance (0 or 1) to the relevance list
#         list_display_relevance.append(groundtruth_dic[image])
        
#         # Depending on the flag, avoid re-clicking already selected images
#         if no_reclick_image:
#             if (groundtruth_dic[image] == 1) and (image not in already_selected_images):
#                 positive_action.append(image)
#             elif (groundtruth_dic[image] == 0) and (image not in already_selected_images):
#                 negatives_action.append(image)
#         else:
#             # If re-clicking is allowed, select all positive or negative images regardless of past selections
#             if groundtruth_dic[image] == 1:
#                 positive_action.append(image)
#             elif groundtruth_dic[image] == 0:
#                 negatives_action.append(image)
    
#     # Select unique prefix-positive actions
#     if k_pos != -1:
#         unique_positive = []
#         used_prefixes = set()
#         for image in positive_action:
#             prefix = image.split("-")[0]
#             if prefix not in used_prefixes:
#                 unique_positive.append(image)
#                 used_prefixes.add(prefix)
#             if len(unique_positive) == k_pos:
#                 break
#         # Fallback: if not enough unique, take first 3 regardless of prefix
#         if len(unique_positive) < k_pos:
#             unique_positive = positive_action[:3]
#         positive_action = unique_positive
    
#     # Select unique prefix-negative actions
#     if k_neg != -1:
#         unique_negative = []
#         used_prefixes = set()
#         for image in negatives_action:
#             prefix = image.split("-")[0]
#             if prefix not in used_prefixes:
#                 unique_negative.append(image)
#                 used_prefixes.add(prefix)
#             if len(unique_negative) == k_neg:
#                 break
#         # Fallback: if not enough unique, take last 3 regardless of prefix
#         if len(unique_negative) < k_neg:
#             unique_negative = negatives_action[-3:]
#         negatives_action = unique_negative

#     # Create tuples of the form (image_name, relevance_label)
#     positive_actions = [(img, 1) for img in positive_action]
#     negative_actions = [(img, 0) for img in negatives_action]
    
#     # Merge actions into a single list
#     actions = positive_actions + negative_actions
#     # print('number of positive actions:', len(positive_actions))
#     # print('positive_action',positive_actions )
#     # print()
#     # print('number of negative actions:', len(negative_actions))
#     # print('negative_action', negative_actions)
#     # Return selected actions and relevance values
#     return actions, list_display_relevance

# def get_actions(display_df_t, groundtruth_dic, already_selected_images, k_pos=-1, k_neg=-1, no_reclick_image=True):
#     # Initialize lists to store positive and negative actions, and the relevance values for displayed images
#     positive_action = []
#     negatives_action = []
#     list_display_relevance = []
    
#     # Iterate through each image in the current display set
#     for image in display_df_t.columns:
#         # Append the ground truth relevance (0 or 1) to the relevance list
#         list_display_relevance.append(groundtruth_dic[image])
        
#         # Depending on the flag, avoid re-clicking already selected images
#         if no_reclick_image:
#             if (groundtruth_dic[image] == 1) and (image not in already_selected_images):
#                 positive_action.append(image)
#             elif (groundtruth_dic[image] == 0) and (image not in already_selected_images):
#                 negatives_action.append(image)
#         else:
#             # If re-clicking is allowed, select all positive or negative images regardless of past selections
#             if groundtruth_dic[image] == 1:
#                 positive_action.append(image)
#             elif groundtruth_dic[image] == 0:
#                 negatives_action.append(image)
                
#     # If a specific number of positive images is requested (k_pos), select them ensuring prefix uniqueness
#     if k_pos != -1:
#         unique_positive = []
#         used_prefixes = set()
#         for image in positive_action:
#             prefix = image.split("-")[0]
#             if prefix not in used_prefixes:
#                 unique_positive.append(image)
#                 used_prefixes.add(prefix)
#             if len(unique_positive) == k_pos:
#                 break
#         # if len(unique_positive) < k_pos:
#         #     print(f"[WARNING] only {len(unique_positive)} img positive , needed {k_pos}.")
#         positive_action = unique_positive
        
#     # If a specific number of negative images is requested (k_neg), select them ensuring prefix uniqueness
#     if k_neg != -1:
#         unique_negative = []
#         used_prefixes = set()
#         for image in negatives_action:
#             prefix = image.split("-")[0]
#             if prefix not in used_prefixes:
#                 unique_negative.append(image)
#                 used_prefixes.add(prefix)
#             if len(unique_negative) == k_neg:
#                 break
#         # if len(unique_negative) < k_neg:
#         #     print(f"[WARNING] only {len(unique_negative)} img negative , needed {k_neg}.")
#         negatives_action = unique_negative
    
#     # Create tuples of the form (image_name, relevance_label) for both positives and negatives
#     positive_actions = [(action, 1) for action in positive_action]
#     negative_actions = [(action, 0) for action in negatives_action]
    
#     # Merge actions into a single list
#     actions = positive_actions + negative_actions

#     # Return selected actions and relevance values for all displayed images
#     return actions, list_display_relevance


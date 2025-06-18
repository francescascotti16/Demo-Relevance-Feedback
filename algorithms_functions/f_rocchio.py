import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from utils.f_display_and_feedback import *
from old_functions.f_evaluation_metrics import *
from utils.functions_similarity_metrics import*
from utils.f_process_data import *
from datetime import datetime



def new_query_rocchio(query_old, relevant, non_relevant, alpha, beta, gamma):
    
    query_old = np.array(query_old).reshape(-1, 1)  # Ensure shape is (1024, 1)
    '''
    Parameters
    ----------
    query_old: old query
    relevant: relevant images
    non_relevant: non relevant images
    alpha: alpha parameter
    beta: beta parameter
    gamma: gamma parameter

    '''
    if relevant.shape[1] == 0:
        centroid_relevant = np.zeros((query_old.shape[0], 1))
    else:
        if len(relevant.shape) == 1:
            centroid_relevant = relevant.reshape(-1, 1)
        else:
            centroid_relevant = np.mean(relevant, axis=1).reshape(-1, 1)
    if non_relevant.shape[1] == 0:
        centroid_non_relevant = np.zeros((query_old.shape[0], 1))
    else:
        if len(non_relevant.shape) == 1:
            centroid_non_relevant = non_relevant.reshape(-1, 1)
        else:
            centroid_non_relevant = np.mean(non_relevant, axis=1).reshape(-1, 1)

    new_query = alpha * query_old + beta * centroid_relevant - gamma * centroid_non_relevant

    return  new_query



def rocchio_single_step(data_df,display_df, relevant_ids,non_relevant_ids, alpha=1, beta=1, gamma=1,fun_name="euclidean", initial_query=None):
    start_time = datetime.now()
    '''
    Parameters
    ----------
    data_df : DataFrame
        DataFrame with the dataset one column for each image 
    display_df : DataFrame with the old display
    relevant_ids : list of indexes of the relevant images 
    non_relevant_ids : list of indexes of the non relevant images
    alpha : float, optional alpha parameter
    beta : float, optional beta parameter
    gamma : float, optional gamma parameter
    fun_name : string, optional function to calculate the similarity or metric model
    initial_query : initial query to start the Rocchio algorithm
    
    '''
    #n_display is the number of columns in the display_df
    n_display=display_df.shape[1]
       
    

    # Initialize the query with zero if not provided
    if initial_query is None:
        old_query = np.zeros((data_df.shape[0], 1))
        # beta=beta+alpha #we want to keep alpha + beta - gamma=1
        # alpha=0
    else:
        old_query=initial_query

    selected_images_at_this_iteration=[im for im in non_relevant_ids]+[im for im in relevant_ids]
    
    if len(selected_images_at_this_iteration)==0:
        return display_df, initial_query
        
    relevant = data_df[relevant_ids].to_numpy()
    non_relevant = data_df[non_relevant_ids].to_numpy() 
    new_query = new_query_rocchio(old_query, relevant, non_relevant, alpha, beta, gamma)
    #old_query = new_query


    if(fun_name=="euclidean") or (fun_name=="triangular") or (fun_name=="jsd") or (fun_name=="sed"):
        distance_vector= get_distance_matrix(pd.DataFrame(new_query), data_df, fun_name=fun_name)
        display_df = create_display(data_df, distance_vector, n_display, is_ascending=True)
    elif(fun_name=="dotproduct") or (fun_name=="cosine"):
        similarity_vector= get_similarity_matrix(pd.DataFrame(new_query), data_df, fun_name=fun_name)
        display_df = create_display(data_df, similarity_vector, n_display, is_ascending=False)
    else:
        print(f"ERR The function {fun_name} is not implemented yet. Using Euclidean distance.")  
        distance_vector= get_distance_matrix(pd.DataFrame(new_query), data_df, fun_name="euclidean")
        display_df = create_display(data_df, distance_vector, n_display, is_ascending=True)
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    return display_df, new_query,elapsed_time
       


def rocchio_single_step_pseudo(data_df,display_df, relevant_ids,non_relevant_ids,fake_non_relevant,fake_relevant, alpha=1, beta=1, gamma=1,fun_name="euclidean", initial_query=None):
    '''
    Parameters
    ----------
    data_df : DataFrame
        DataFrame with the dataset one column for each image 
    display_df : DataFrame with the old display
    relevant_ids : list of indexes of the relevant images 
    non_relevant_ids : list of indexes of the non relevant images
    alpha : float, optional alpha parameter
    beta : float, optional beta parameter
    gamma : float, optional gamma parameter
    fun_name : string, optional function to calculate the similarity or metric model
    initial_query : initial query to start the Rocchio algorithm
    
    '''
    #n_display is the number of columns in the display_df
    n_display=display_df.shape[1]
       
    

    # Initialize the query with zero if not provided
    if initial_query is None:
        old_query = np.zeros((data_df.shape[0], 1))
        # beta=beta+alpha #we want to keep alpha + beta - gamma=1
        # alpha=0
    else:
        old_query=initial_query

    selected_images_at_this_iteration=[im for im in non_relevant_ids]+[im for im in relevant_ids]
    
    if len(selected_images_at_this_iteration)==0:
        return display_df, initial_query
    
    relevant = data_df[relevant_ids].to_numpy()
    
    
    relevant= np.concatenate([relevant, fake_relevant], axis=1)

    non_relevant = data_df[non_relevant_ids].to_numpy() 
    non_relevant= np.concatenate([non_relevant, fake_non_relevant], axis=1)
    new_query = new_query_rocchio(old_query, relevant, non_relevant, alpha, beta, gamma)
    #old_query = new_query


    if(fun_name=="euclidean") or (fun_name=="triangular") or (fun_name=="jsd") or (fun_name=="sed"):
        distance_vector= get_distance_matrix(pd.DataFrame(new_query), data_df, fun_name=fun_name)
        display_df = create_display(data_df, distance_vector, n_display, is_ascending=True)
    elif(fun_name=="dotproduct") or (fun_name=="cosine"):
        similarity_vector= get_similarity_matrix(pd.DataFrame(new_query), data_df, fun_name=fun_name)
        display_df = create_display(data_df, similarity_vector, n_display, is_ascending=False)
    else:
        print(f"ERR The function {fun_name} is not implemented yet. Using Euclidean distance.")  
        distance_vector= get_distance_matrix(pd.DataFrame(new_query), data_df, fun_name="euclidean")
        display_df = create_display(data_df, distance_vector, n_display, is_ascending=True)

    return display_df, new_query
       


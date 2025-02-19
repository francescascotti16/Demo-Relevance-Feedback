
import numpy as np
import pandas as pd
from utils.functions_similarity_metrics import get_distance_matrix, get_similarity_matrix
from utils.f_display_and_feedback import create_display


from datetime import datetime


def new_query_rocchio(query_old, relevant, non_relevant, alpha, beta, gamma):
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
    start_time = datetime.now()
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
    end_time = datetime.now()
    return  new_query, end_time-start_time



def rocchio_single_step(data_df,display_df, relevant_ids,non_relevant_ids, alpha=1, beta=1, gamma=1,fun_name="euclidean", initial_query=None):
    
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
    start_time = datetime.now()
    #n_display is the number of columns in the display_df
    n_display=display_df.shape[1]
       
    old_query=initial_query

    # Initialize the query with zero if not provided
    if initial_query is None:
        old_query = np.zeros((data_df.shape[0], 1))
        # beta=beta+alpha #we want to keep alpha + beta - gamma=1
        # alpha=0


    #selected_images_at_this_iteration=[im for im in non_relevant_ids]+[im for im in relevant_ids]
    selected_images_at_this_iteration = non_relevant_ids + relevant_ids

    if len(selected_images_at_this_iteration)==0:
        return display_df, initial_query
        
    relevant = data_df[relevant_ids].to_numpy()
    non_relevant = data_df[non_relevant_ids].to_numpy() #FIXME if data_df[non_relevant_ids] null 
    
    new_query, time_new_query = new_query_rocchio(old_query, relevant, non_relevant, alpha, beta, gamma)
    old_query = new_query


    if(fun_name=="euclidean") or (fun_name=="triangular") or (fun_name=="jsd") or (fun_name=="sed"):
        distance_vector, get_distance_matrix_time= get_distance_matrix(pd.DataFrame(new_query), data_df, fun_name=fun_name)
        display_df, time_create_display= create_display(data_df, distance_vector, n_display, is_ascending=True)
    elif(fun_name=="dotproduct") or (fun_name=="cosine"):
        similarity_vector, get_distance_matrix_time= get_similarity_matrix(pd.DataFrame(new_query), data_df, fun_name=fun_name)
        display_df,time_create_display = create_display(data_df, similarity_vector, n_display, is_ascending=False)
    else:
        print(f"ERR The function {fun_name} is not implemented yet. Using Euclidean distance.")  
        distance_vector,get_distance_matrix_time= get_distance_matrix(pd.DataFrame(new_query), data_df, fun_name="euclidean")
        display_df,time_create_display = create_display(data_df, distance_vector, n_display, is_ascending=True)
    end_time = datetime.now()
    total_time_single_step=end_time-start_time
    return display_df, new_query, time_new_query, total_time_single_step, get_distance_matrix_time, time_create_display
       


import numpy as np
import pandas as pd
from functions_similarity_metrics import compute_similarity_score
from f_display_and_feedback import create_display
import numpy as np
from scipy.special import softmax as sp_softmax
from functions_similarity_metrics import*
from datetime import datetime


def polyquery_score(old_scores, data_df, relevant, non_relevant, alpha, beta, gamma,fun_name):
    '''
    Parameters
    ----------
    old_scores: old scores
    relevant: relevant images
    non_relevant: non relevant images
    alpha: alpha parameter
    beta: beta parameter
    gamma: gamma parameter
    fun_name: function to calculate the similarity scores

    '''
    
    #if relevant is null we set centroid relevant to be a vector of 0 and the same for non relevant
    if relevant.shape[1] == 0:
        relevant_scores = np.zeros((old_scores.shape[0], 1))
       # print(f"relevant_scores shape: {relevant_scores.shape}")
    else:
        if len(relevant.shape) == 1:
            centroid_relevant = relevant.reshape(-1, 1)
        else:
            centroid_relevant = np.mean(relevant, axis=1).reshape(-1, 1)
    
        relevant_scores, total_time_sed, complexity_time_avg, complexity_time_x, complexity_time_y, total_time=compute_similarity_score(centroid_relevant,data_df, fun_name=fun_name)

    if non_relevant.shape[1] == 0:
        non_relevant_scores = np.zeros((old_scores.shape[0], 1))
    else:
        if len(non_relevant.shape) == 1:
            centroid_non_relevant = non_relevant.reshape(-1, 1)
        else:
            centroid_non_relevant = np.mean(non_relevant, axis=1).reshape(-1, 1)
        non_relevant_scores, total_time_sed2, complexity_time_avg2, complexity_time_x2, complexity_time_y2, total_time2=compute_similarity_score(centroid_non_relevant,data_df, fun_name=fun_name)
    
    new_scores = alpha * old_scores.reshape(-1, 1) + beta * relevant_scores - gamma * non_relevant_scores
    
   
    return  new_scores, total_time_sed, complexity_time_avg, complexity_time_x, complexity_time_y, total_time, total_time_sed2, complexity_time_avg2, complexity_time_x2, complexity_time_y2, total_time2


def poly_single_step(data_df,display_df, relevant_ids,non_relevant_ids, alpha=1, beta=0.7, gamma=0.7,fun_name="sed", initial_query=None, initial_scores=None):
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
    
    old_scores=initial_scores
    # Initialize the query with zero if not provided
    if old_scores is None:
        if initial_query is None:
            old_scores= np.array([0] * data_df.shape[1]) 
            # beta=beta+alpha #we want to keep alpha + beta - gamma=1
            # alpha=0
        else:
            old_scores = compute_similarity_score(initial_query,data_df, fun_name=fun_name) 

    selected_images_at_this_iteration=[im for im in non_relevant_ids]+[im for im in relevant_ids]
    
    if len(selected_images_at_this_iteration)==0:
        return display_df, old_scores
        
    relevant = data_df[relevant_ids].to_numpy()
    non_relevant = data_df[non_relevant_ids].to_numpy() #FIXME if data_df[non_relevant_ids] null 
    new_scores, total_time_sed, complexity_time_avg, complexity_time_x, complexity_time_y, total_time, total_time_sed2, complexity_time_avg2, complexity_time_x2, complexity_time_y2, total_time2= polyquery_score(old_scores, data_df, relevant, non_relevant, alpha, beta, gamma,fun_name=fun_name)
    display_df ,time= create_display(data_df, new_scores, n_display, is_ascending=False)
    
    end_time = datetime.now()
    total_time_single_step = end_time - start_time

    return display_df, new_scores, total_time_sed, complexity_time_avg, complexity_time_x, complexity_time_y, total_time, total_time_sed2, complexity_time_avg2, complexity_time_x2, complexity_time_y2, total_time2 ,total_time_single_step

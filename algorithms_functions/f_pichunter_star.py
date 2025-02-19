from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from utils.functions_similarity_metrics import  get_similarity_matrix_temperature
from utils.f_display_and_feedback import create_display
from datetime import datetime

def pichunter_single_step(data_df,display_df, relevant_ids,non_relevant_ids,fun_name="softmin", initial_prob=0, temperature=1):
    start_time = datetime.now()
    '''
    Parameters
    ----------
    data_df: DataFrame with the dataset, one column for each image
    display_df: DataFrame with the display, one column for each image
    relevant_ids: list of relevant images
    non_relevant_ids: list of non relevant images
    fun_name: function to calculate the user model
    initial_prob: initial probability of each image being relevant
    temperature: temperature parameter for the softmax/min function
    
    Returns
    -------
    display_df: DataFrame with the new display
    new_prob_values: new probability values of each image being relevant
    
    
    '''
    n_display=display_df.shape[1]

    if (not isinstance(initial_prob, np.ndarray)) or (initial_prob is None):
        initial_prob =  np.full(data_df.shape[1],0.5) # prob of each image being relevant is 0.5
    

    # Calculate similarity and sissimilarity matrix
    similarity_matrix, time_get_similarity =get_similarity_matrix_temperature(display_df, data_df, user_model_fun_name=fun_name, temperature=temperature)
    dissimilarity_matrix=1/similarity_matrix #np.where(similarity_matrix!=0, 1/similarity_matrix, 1)
         
    #compute the soft similarity and dissimilarity matrix
    soft_similarity_matrix=similarity_matrix/similarity_matrix.sum(axis=1)[:, np.newaxis] 
    soft_dissimilarity_matrix=dissimilarity_matrix/dissimilarity_matrix.sum(axis=1)[:, np.newaxis]
    
    #create the dataframes with the soft similarity and dissimilarity matrices
    similarity_df = pd.DataFrame(soft_similarity_matrix, columns=display_df.columns, index=data_df.columns) #similarities to display images
    dissimilarity_df = pd.DataFrame(soft_dissimilarity_matrix, columns=display_df.columns, index=data_df.columns) #dissimilarities to display images
        
    # Calculate user model
    
    image_in_display=display_df.columns
    relevant_ids_in_display=[im for im in relevant_ids if im in image_in_display]
    non_relevant_ids_in_display=[im for im in non_relevant_ids if im in image_in_display]


    p_relevant=initial_prob #p(oi \in T, H_{t-1})
 

    prod_similartities_to_positive_actions= np.array(similarity_df[relevant_ids_in_display].product(axis=1))#similarity of each image to images selected as positive
    prod_dissimilarities_to_negative_actions=1
    if len(non_relevant_ids_in_display)>0:
        prod_dissimilarities_to_negative_actions= np.array(dissimilarity_df[non_relevant_ids_in_display].product(axis=1)) #dissimilarity of each image to images selected as negative

    usermodel_given_relevant= prod_similartities_to_positive_actions*prod_dissimilarities_to_negative_actions# (product over postive actions of dimilarities  * product over negative actions of dissim)
   
    numerator=usermodel_given_relevant*p_relevant
    denominator=numerator.sum()


    new_prob_values = numerator / denominator 
    display_df ,time_display= create_display(data_df, new_prob_values, n_display)
    end_time = datetime.now()
    total_time_single_step=end_time-start_time
    return display_df, new_prob_values, total_time_single_step, time_display, time_get_similarity

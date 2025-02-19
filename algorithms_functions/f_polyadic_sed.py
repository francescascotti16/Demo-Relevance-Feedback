import numpy as np
import pandas as pd


from utils.f_display_and_feedback import create_display
from old_functions.f_evaluation_metrics import calculate_ap_for_iterations, calculate_ndcg_for_iterations, calculate_recall_at_k_for_iterations
from datetime import datetime
from utils.functions_similarity_metrics import shannon_entropy
def get_sed_logscale_sim(data_df,entropy_dict,relevant_ids,non_relevant_ids,old_score,alpha,beta,gamma,dict):
    
    ''' 
    Parameters:
    data_df: DataFrame with the dataset one column for each image
    entropy_dict: dictionary with the entropy of each image, image_id are the key
    relevant_ids: list of relevant images
    non_relevant_ids: list of non relevant images
    beta: beta parameter
    gamma: gamma parameter
        dict : dictionary with the precomputed values
        precomputed_sum_pos: precomputed sum of the positive images + data
        precomputed_sum_neg: precomputed sum of the negative images +data
        precomputed_entropy_avg_neg: precomputed sum of the entropy of the negative images and the data complexity
        precomputed_entropy_avg_pos: precomputed sum of ethentropy  of the positive images and the data complexity
    Returns:
    new_values: list of new values
    '''     
    start_time = datetime.now()
    sed_score=[] 

    precomputed_sum_neg=dict['precomputed_sum_neg']
    precomputed_sum_pos=dict['precomputed_sum_pos']
    precomputed_entropy_avg_neg=dict['precomputed_entropy_avg_neg']
    precomputed_entropy_avg_pos=dict['precomputed_entropy_avg_pos']
    n_pos=dict['n_pos']
    n_neg=dict['n_neg']
   

    for el in relevant_ids:
        numpy_data=data_df[el].to_numpy().reshape(1, -1)
        precomputed_sum_pos+=numpy_data
        n_pos+=1
    precomputed_entropy_avg_pos=shannon_entropy((precomputed_sum_pos/n_pos) )[0]
    


    for el in non_relevant_ids: 
        numpy_data=data_df[el].to_numpy().reshape(1, -1)
        precomputed_sum_neg+=numpy_data
        n_neg+=1
    precomputed_entropy_avg_neg=shannon_entropy((precomputed_sum_neg/n_neg) ) [0]
    
    new_dict={'precomputed_sum_pos':precomputed_sum_pos,
              'precomputed_sum_neg':precomputed_sum_neg,
              'precomputed_entropy_avg_neg':precomputed_entropy_avg_neg,
              'precomputed_entropy_avg_pos':precomputed_entropy_avg_pos,
              'n_pos':n_pos,
              'n_neg':n_neg}
            
    #iterate over data_df
     
    for img_id, row in data_df.T.iterrows():
        data_vec=row.to_numpy().reshape(1, -1)
        data_entropy=entropy_dict[img_id]  
        
        mean_pos=((precomputed_sum_pos/n_pos)+data_vec)/2.0
        entropy_numerator_pos=shannon_entropy(mean_pos)[0]
        
        
        avg_entropy_denominator_pos=(precomputed_entropy_avg_pos+data_entropy)/2.0
        score_pos= 2-np.exp(entropy_numerator_pos-avg_entropy_denominator_pos)

        mean_neg=((precomputed_sum_neg/n_neg)+data_vec)/2.0
        entropy_numerator_neg=shannon_entropy(mean_neg)[0]
        avg_entropy_denominator_neg=(precomputed_entropy_avg_neg+data_entropy)/2.0
        score_neg= 2-np.exp(entropy_numerator_neg-avg_entropy_denominator_neg)


        score= beta*score_pos-gamma*score_neg
        sed_score.append(score)
        

    new_scores=alpha*old_score+np.array(sed_score).flatten()
    end_time = datetime.now()
    total_time = end_time - start_time
    return new_scores, new_dict, total_time
    

def get_sed_logscale_sim_vec(data_df,entropy_dict,query):
    '''
    Parameters:
    data_df: DataFrame with the dataset one column for each image
    entropy_dict: dictionary with the entropy of each image, image_id are the key
    query: query vector
    Returns:
    sed_values: list of new values

    '''
    # m = df_with_complexity.shape[0]
    # sed_values = np.zeros((m, 1))       

    sed_score=[] 

    query_complexity= np.exp(shannon_entropy(query))[0]# complexity 

    for index, row in data_df.iterrows():
        data_vec=row.to_numpy().reshape(1, -1)
        sum_pos= query+data_vec
        mean_pos=sum_pos/(2.0)
        complexity_pos_num=np.exp(shannon_entropy(mean_pos))[0]
        complexity_pos_product_den=query_complexity*np.exp((entropy_dict[index]))
        score_pos= 2- complexity_pos_num/ np.sqrt(complexity_pos_product_den)
   
        sed_score.append(score_pos)
        print(f"score_pos: {score_pos} type: {type(score_pos)}")

    return np.array(sed_score).flatten()


def poly_sed_logscale_single_step(data_df,display_df, relevant_ids,non_relevant_ids,precomputed_dict_initial=None, alpha=0.7, beta=0.7, gamma=0.4, initial_query=None, initial_scores=None,entropy_dict=None):
    '''
    Parameters
    ----------
    data_df : DataFrame
        DataFrame with the dataset one column for each image
    display_df : DataFrame
        DataFrame with the old display one column for each image
    relevant_ids : list of relevant images
    non_relevant_ids : list of non relevant images
    precomputed_dict_initial : dictionary with the precomputed values
    alpha : float, optional
        DESCRIPTION. The default is 0.7.
    beta : float, optional
        DESCRIPTION. The default is 0.7.
    gamma : float, optional
        DESCRIPTION. The default is 0.4.
    initial_query : TYPE, optional
        DESCRIPTION. The default is None.
    initial_scores : TYPE, optional 
        DESCRIPTION. The default is None.
    entropy_dict : dictionary, optional
        dictionary with the entropy of each image, image_id are the key
    Returns
    -------
    display_df : DataFrame
        new diplay
    new_scores : list
        new scores for each image
    precomputed_dict : dictionary
        updated dictionary with the precomputed values
    entropy_dict : dictionary
        dictionary with the entropy of each image

    '''
    start_time = datetime.now()
    #n_display is the number of columns in the display_df
    n_display=display_df.shape[1]


    if entropy_dict is None:
       #create a dataframe with the complexity of each image
        entropy_dict={}
        for index, row in data_df.T.iterrows():
            #data_complexity_df.loc[index,'data']= row.to_numpy().reshape(1, -1)
            entropy_dict[index]=shannon_entropy(row)[0]
        

    if precomputed_dict_initial is None:
        precomputed_dict={"precomputed_sum_pos":0, "precomputed_sum_neg":0, 
                        "precomputed_entropy_avg_neg":1,
                        "precomputed_entropy_avg_pos":1,
                        "n_pos":0, 
                        "n_neg":0}
    else:
        precomputed_dict=precomputed_dict_initial.copy()
    
    old_scores=initial_scores
    # Initialize the query with zero if not provided
    if old_scores is None:
        if initial_query is None:
            old_scores= np.array([0] *data_df.shape[1] ) 
            # beta=beta+alpha #we want to keep alpha + beta - gamma=1
            # alpha=0
        else:
            old_scores = get_sed_logscale_sim_vec(data_df,entropy_dict,initial_query)

    selected_images_at_this_iteration=[im for im in non_relevant_ids]+[im for im in relevant_ids]

    if len(selected_images_at_this_iteration)==0:
        return display_df, old_scores,precomputed_dict,entropy_dict
   
    
              
    new_scores, precomputed_dict, time_get_sed_logscale= get_sed_logscale_sim(data_df,entropy_dict,relevant_ids,non_relevant_ids,old_scores,alpha,beta,gamma,precomputed_dict)
    display_df, time_display = create_display(data_df, new_scores, n_display, is_ascending=False)
    end_time = datetime.now()   
    total_time = end_time - start_time
    return display_df, new_scores, precomputed_dict, entropy_dict, total_time,time_get_sed_logscale,time_display

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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime

def distance_from_hyperplane(w, b, data_df):
    return np.dot(data_df.T, w) + b

def svm_score(data_df, relevant, non_relevant):
    '''
    Parameters
    ----------
    relevant: relevant images
    non_relevant: non relevant images
    '''

    # X  Training vectors, i.e. teh union of relevant and non_relevant, y=Target values (class labels in classification='1' for relevant  or '0' for non-relevant).
    X = np.concatenate((relevant, non_relevant), axis=0)
    y = np.concatenate((np.ones(relevant.shape[0]), np.zeros(non_relevant.shape[0])))
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    accuracy = accuracy_score(y, clf.predict(X))
    distances_to_hyper = distance_from_hyperplane(w, b, data_df)
    
    return  distances_to_hyper



def svm_single_step(data_df,display_df, relevant_ids,non_relevant_ids,initial_scores=None,alpha=0, beta=1,query=None):
    star_time = datetime.now()
    n_display=display_df.shape[1]
 
    
    old_scores=initial_scores
    # Initialize the query with zero if not provided
    if old_scores is None:
        old_scores= (np.array([0] * data_df.shape[1]) )

    if len(relevant_ids)==0: 
         return display_df,old_scores

    
    # print('number of relevant images:', len(relevant_ids))
    # print('number of non relevant images:', len(non_relevant_ids))
    relevant = data_df[relevant_ids].to_numpy()
    # print("relevant shape:", relevant.shape)        # Should be (n_features, n_samples), e.g., (13, N)
    # print("query shape before reshape:", query.shape)

    query = query.reshape(-1, 1)

# Concatenate along the second axis (columns/samples)
    relevant_with_q = np.concatenate([relevant, query], axis=1).T
    non_relevant = data_df[non_relevant_ids].to_numpy().T
   
    new_scores = alpha*old_scores+beta*svm_score(data_df, relevant_with_q, non_relevant)
    display_df = create_display_svm(data_df, new_scores, n_display, is_ascending=False)
    # remouve from display df the images that comes from the external negatives
    end_time = datetime.now()
    elapsed_time = end_time - star_time
    return display_df, new_scores, elapsed_time
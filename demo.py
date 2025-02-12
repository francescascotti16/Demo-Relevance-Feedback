
# libraries 
from flask import Flask, request, jsonify, send_from_directory
import json
import requests
import pandas as pd
import pickle
import tqdm
import time 

#general functions

from f_evaluation_metrics import*
from f_display_and_feedback import *
from f_process_data import *

# algortihms functions
from f_rocchio import *

from f_pichunter_star import *
from f_svm import *
from functions_similarity_metrics import *
from f_polyquery_msed_logscale import *
from function_polyadic_new import *

import time
import requests
import json
import pandas as pd
from flask import request, jsonify

#indexed data functions 
from f_interfaccia import *

#algorithms functions for renaming 

from f_pichunter_star import pichunter_single_step as pichunter_single_step_star
from datetime import datetime
import hashlib
import json
import pandas as pd
import numpy as np
from f_display_and_feedback import *
from f_evaluation_metrics import *
from f_pichunter_star import *

from f_polyquery_msed_logscale import *
from f_polyquery_sed_logscale import *
from f_process_data import *
from f_rocchio import *
from f_svm import *
from f_pichunter import *
from f_process_data import *

#Initialize query_value_rocchio globally

import pickle
from tqdm import tqdm
app = Flask(__name__)

# Definizione dei percorsi dei file e nomi delle variabili
file_paths = {
    '/home/francescascotti/dev/interfaccia/index_2_id.pkl': 'index_2_id',
    '/home/francescascotti/dev/interfaccia/indexed_data': 'indexed_data',
    '/home/francescascotti/dev/interfaccia/indexed_ids': 'indexed_ids',
    '/home/francescascotti/dev/interfaccia/indexed_data_logistic': 'indexed_data_logistic'
}
##
# Lista per tenere traccia dei dati caricati
loaded_data = {}

# Utilizza tqdm per monitorare il progresso
for file_path, var_name in tqdm(file_paths.items(), desc='Caricamento file di dati'):
    with open(file_path, 'rb') as f:
        loaded_data[var_name] = pickle.load(f)

# Assegnazione delle variabili
index_2_id = loaded_data['index_2_id']
indexed_data = loaded_data['indexed_data']
indexed_ids = loaded_data['indexed_ids']
indexed_data_logistic = loaded_data['indexed_data_logistic']



query_value_rocchio = None


new_prob_values_pichunter_star=None
new_prob_values_pic=None
score_value_polyadic= None
score_value_polyquery_msed=None
complexity_dict_value= None
precomputed_dict_value=None
precomputed_dict_polyquery_msed_log_value=None
entropy_dict_value=None
score_value_polyquery_msed_log=None
score_value_polyadic_jsd=None


@app.route('/')
def index():
    return send_from_directory('static', 'demo.html')


@app.route('/search', methods=['POST'])
def search(n_display=400):
    global data_df, df_display, data_df_log, df_display_log , query_features_total

    
    
    data = request.get_json()
    query_orig = data['query']

    
    host = "https://visione.isti.cnr.it"
    textual_mode = "clip-laion"
    max_rank =1000

    query = json.dumps({"query": [{"textual": query_orig}], "parameters": [{"textualMode": textual_mode, "occur": "and", "simReorder": "false"}]})
    query_features_total=fetch_text_feature(query)
   
   
    
    results = requests.post(host + '/services/core/search', data={'query': query, 'sortbyvideo': False, 'maxres': max_rank}, verify=False)
  
    df_results = pd.DataFrame(results.json())

    
    img_ids = df_results['imgId'][:n_display].tolist()
  
    data_df = create_dataframe_from_results(df_results, 
                                            index_2_id, 
                                            indexed_data, 
                                            indexed_ids)
    
    df_display_col_names = df_results['imgId'].head(400).tolist()
    df_display = data_df[df_display_col_names]
    
    data_df_log = create_dataframe_from_results(df_results, 
                                                index_2_id, 
                                                indexed_data_logistic, 
                                                indexed_ids)
   
    
    df_display_col_names_log = df_results['imgId'].head(400).tolist()
    df_display_log = data_df_log[df_display_col_names_log]
    print('shape of data_df_log:', data_df_log.shape)
    image_urls = ["https://visione.isti.cnr.it/frames/{}/{}.png".format(img_id.split('-')[0], img_id) for img_id in img_ids]
  

    return jsonify({'image_urls': image_urls, 'img_ids': img_ids})


@app.route('/save_and_update', methods=['POST'])
def save_and_update():
    global query_value_rocchio,new_prob_values_pichunter_star, score_value_polyadic,score_value_polyadic_jsd, score_value_polyquery_msed, complexity_dict_value, precomputed_dict_value, precomputed_dict_polyquery_msed_log_value, entropy_dict_value, score_value_polyquery_msed_log
    global data_df, df_display, data_df_log, df_display_log, time_of_search,new_prob_values_pic
    
    data = request.get_json()
    relevant_image_ids = data.get('relevant_images_ids', [])
    non_relevant_image_ids = data.get('non_relevant_images_ids', [])
    relevant_image_ids_temp= data.get('relevant_images_ids_temp',[])
    non_relevant_image_ids_temp = data.get('non_relevant_images_ids_temp', [])
    
    selected_algorithm = data.get('selected_algorithm', 'Rocchio').lower()  # Default to Rocchio if not specified
    
    # Save relevant and non-relevant image IDs in JSON files
    with open('relevant_images_ids.json', 'w') as f_relevant:
        json.dump({'relevant_images_ids': relevant_image_ids}, f_relevant)

    with open('non_relevant_images_ids.json', 'w') as f_non_relevant:
        json.dump({'non_relevant_images_ids': non_relevant_image_ids}, f_non_relevant)
   
   
    # Execute the selected algorithm
    
    if selected_algorithm == 'rocchio':
        # Use query_value_rocchio as initial_query
  
        df_display, new_query, time_new_query, time_of_search, get_distance_matrix_time, time_create_display = rocchio_single_step(data_df, df_display,
                                                    relevant_image_ids_temp, non_relevant_image_ids_temp, 
                                                    alpha=0.75, beta=1, gamma=0.75, 
                                                    fun_name="euclidean", 
                                                    initial_query=query_value_rocchio)
        query_value_rocchio=new_query
        # Update query_value_rocchio for next iteration
        

        
    elif selected_algorithm == 'pichunter-star':
        # eliminate the non_relevant_image_ids that are not in df_display_columns:data_df,display_df, relevant_ids,fun_name="softmin", initial_prob=0,temperature=1)
              
        df_display, new_prob_values_star, time_of_search, time_display, time_get_similarityx = pichunter_single_step_star(data_df, df_display, relevant_image_ids_temp, 
                                                            non_relevant_image_ids_temp, 
                                                            fun_name="softmin", 
                                                            initial_prob=new_prob_values_pichunter_star,temperature=82.10553)
       
        new_prob_values_pichunter_star = new_prob_values_star
    elif selected_algorithm == 'pichunter':
        non_relevant_image_ids=[]
        df_display, new_prob_values ,time_of_search =pichunter_single_step_old(data_df,df_display, relevant_image_ids_temp,fun_name="softmin", initial_prob=new_prob_values_pic,temperature=1)
        new_prob_values_pic = new_prob_values
    elif selected_algorithm == 'svm':
        df_display, new_scores, time_of_search, time_distance_from_hyperplane, total_time_score_computation, total_time_display = svm_single_step(data_df, df_display, relevant_image_ids, non_relevant_image_ids)
        
        
   
    elif selected_algorithm == 'polyadic-sed':
        
        df_display, new_scores, _, _, _, _, _, _, _,_, _, _,time_of_search=poly_single_step(data_df_log,df_display_log, relevant_image_ids,non_relevant_image_ids,alpha=0.75, 
                                                                                              beta=1, gamma=0.75,fun_name="sed", initial_query=None, initial_scores=score_value_polyadic)

        score_value_polyadic = new_scores
        
        
       
    elif selected_algorithm == 'polyadic-msed':
        df_display, new_scores, precomputed_dict, entropy_dict ,time_of_search,_,_= poly_msed_logscale_single_step(data_df_log, df_display_log, 
                                                                                                relevant_image_ids, 
                                                                                                non_relevant_image_ids, 
                                                                                                precomputed_dict_initial=precomputed_dict_polyquery_msed_log_value, 
                                                                                                alpha=0.7, beta=0.7, gamma=0.4, 
                                                                                                initial_query=None, 
                                                                                                initial_scores=score_value_polyquery_msed_log,
                                                                                                entropy_dict=entropy_dict_value)
        score_value_polyquery_msed_log = new_scores
        precomputed_dict_polyquery_msed_log_value=precomputed_dict
        entropy_dict_value=entropy_dict
        #print error message if the algorithm is not recognized
        print("Error: Algorithm not recognized")
        # use Rocchio 
        
        # Use query_value_rocchio as initial_query
    else:
        #print error message if the algorithm is not recognized
        print("Error: Algorithm not recognized")
        
        df_display, new_query, time_new_query, time_of_search, get_distance_matrix_time, time_create_display = rocchio_single_step(data_df, df_display,
                                                    relevant_image_ids_temp, non_relevant_image_ids_temp, 
                                                    alpha=0.75, beta=1, gamma=0.75, 
                                                    fun_name="euclidean", 
                                                    initial_query=query_value_rocchio)
        query_value_rocchio=new_query
        # Update query_value_rocchio for next iteration


  

    # Update the list of images to be displayed
    new_img_ids = df_display.columns.tolist()
    new_image_urls = [f"https://visione.isti.cnr.it/frames/{img_id.split('-')[0]}/{img_id}.png" for img_id in new_img_ids]
   # empty non_relevant_image_ids
    print(f"Non relevant image ids : {non_relevant_image_ids}")
    print(f"Relevant image ids : {relevant_image_ids}")
    print(f"Non relevant image ids temp : {non_relevant_image_ids_temp}")
    print(f"Relevant image ids  temp: {relevant_image_ids_temp}")
    print(f'time of search {time_of_search}')
    # Prepare response with relevant data
    time_of_search=str(time_of_search.total_seconds()) + " seconds"
    print(f"Final time_of_search: {time_of_search}")

    response_data = {
        'status': 'success',
        'relevant_images_ids': relevant_image_ids,
        'non_relevant_images_ids': non_relevant_image_ids,
        'relevant_images_ids_temp': relevant_image_ids_temp,
        'non_relevant_images_ids_temp': non_relevant_image_ids_temp,
        'selected_algorithm': selected_algorithm,
        'new_image_ids': new_img_ids,  # Return new image IDs
        'new_image_urls': new_image_urls,  # Return new image URLs
        'total_time': time_of_search
    }

    return jsonify(response_data)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000) 
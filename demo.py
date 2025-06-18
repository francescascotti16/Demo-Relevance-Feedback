# Libraries 
from flask import Flask, request, jsonify, send_from_directory
import json
import requests
import pandas as pd
import pickle
import tqdm
import uuid
from collections import deque
# General functions
import numpy as np
from utils.f_process_data import *
from utils.f_display_and_feedback import *

# Algorithms functions
from algorithms_functions.f_rocchio import *
from algorithms_functions.f_pichunter_star import *
from algorithms_functions.f_svm import *
from utils.functions_similarity_metrics import *
from algorithms_functions.f_polyquery_msed_logscale import *
from algorithms_functions.f_polyadic_sed import *
from utils.demo_functions import fetch_text_feature, create_dataframe_from_results, compute_first_probability

# Indexed data functions 
from utils.f_files import *

# Algorithms functions for renaming
from algorithms_functions.f_pichunter_star import pichunter_single_step as pichunter_single_step_star

#Initialize query_value_rocchio globally

import pickle
from tqdm import tqdm
app = Flask(__name__)

# Buffer circolare con massimo 5 sessioni
MAX_SESSIONS = 10
sessions = deque(maxlen=MAX_SESSIONS)
session_data = {}  # Dizionario per gestire i dati di ogni sessione


file_paths = {
    '/home/francescascotti/dev/interfaccia/index_2_id.pkl': 'index_2_id',
    '/home/francescascotti/dev/interfaccia/indexed_data': 'indexed_data',
    '/home/francescascotti/dev/interfaccia/indexed_ids': 'indexed_ids',
    '/home/francescascotti/dev/interfaccia/indexed_data_logistic': 'indexed_data_logistic'
}
##
loaded_data = {}

for file_path, var_name in tqdm(file_paths.items(), desc='Caricamento file di dati'):
    with open(file_path, 'rb') as f:
        loaded_data[var_name] = pickle.load(f)

index_2_id = loaded_data['index_2_id']
indexed_data = loaded_data['indexed_data']
indexed_ids = loaded_data['indexed_ids']
indexed_data_logistic = loaded_data['indexed_data_logistic']



query_value_rocchio = None


query_value_rocchio_log= None
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
precomputed_dict_polyquery_sed_log_value=None
entropy_dict_value_sed=None
score_value_polyquery_sed_log=None
initial_prob_pichunter = None
@app.route('/')
def index():
    return send_from_directory('static', 'demo.html')

display_number=400
@app.route('/search', methods=['POST'])
def search(n_display=display_number):
    global data_df, df_display, data_df_log, df_display_log , query_features_total
    global sessions

    
    
    data = request.get_json()
    query_orig = data['query']

    session_id = str(uuid.uuid4())
    if len(sessions) >= MAX_SESSIONS:
        removed_session = sessions.popleft()
        print(f"Session removed: {removed_session}")
    sessions.append(session_id)
    print(f"New session added: {session_id}")
    session_data[session_id] = {
        "query_value_rocchio": None,
        "query_value_rocchio_log": None,
        "query_value_decap": None,
        "new_prob_values_pichunter_star": None,
        "new_prob_values_pic": None,
        "score_value_polyadic": None,
        "score_value_polyquery_msed": None,
        "complexity_dict_value": None,
        "precomputed_dict_value": None,
        "precomputed_dict_polyquery_msed_log_value": None,
        "entropy_dict_value": None,
        "score_value_polyquery_msed_log": None,
        "score_value_polyadic_jsd": None,
        "precomputed_dict_polyquery_sed_log_value": None,
        "entropy_dict_value_sed": None,
        "score_value_polyquery_sed_log": None,
        "query_value_svm": None,
        "initial_prob_pichunter": None,
    }
    # new session
    
    host = "https://visione.isti.cnr.it"
    textual_mode = "clip-laion"
    max_rank =10000
    max_shuffle = 10000
    query = json.dumps({"query": [{"textual": query_orig}], "parameters": [{"textualMode": textual_mode, "occur": "and", "simReorder": "false"}]})
   
    
    results = requests.post(host + '/services/core/search', data={'query': query, 'sortbyvideo': False, 'maxres': max_rank})
    data = results.json()
    initial_query_emb = np.array(fetch_text_feature(query), dtype=np.float64)
    initial_query_emb = np.squeeze(initial_query_emb).reshape(-1, 1)
    initial_query_emb_log = logistic(initial_query_emb)

    # Save the JSON data to a file called 'results.json'
    with open("results.json", "w") as outfile:
        json.dump(data, outfile, indent=4)
        
    df_results = pd.DataFrame(results.json())

    
    #img_ids = df_results['imgId'][:n_display].tolist()
    
    data_df = create_dataframe_from_results(df_results, 
                                            index_2_id, 
                                            indexed_data, 
                                            indexed_ids)
   
                    
    
       
   
    
    df_display_col_names = df_results['imgId'].head(display_number).tolist()
    df_display = data_df[df_display_col_names]
   
    data_df_log = create_dataframe_from_results(df_results, 
                                                index_2_id, 
                                                indexed_data_logistic, 
                                                indexed_ids)
    
   

    img_ids= data_df.columns.tolist()[:n_display]
    
    # Calcolo entropy_dict (solo se non precomputato)
    entropy_dict_log = {img_id: shannon_entropy(data_df_log[img_id]) for img_id in data_df_log.columns}

    # Calcolo di rho₀ = s_poly(q0, oi)
    rho0_polyquery_msed_log = get_msed_logscale_sim_vec(data_df_log, entropy_dict_log, initial_query_emb_log)
    # Calcolo entropia per Polyadic-SED
    entropy_dict_log_sed = {img_id: shannon_entropy(data_df_log[img_id]) for img_id in data_df_log.columns}

    # Calcolo rho₀ (score iniziale)
    rho0_polyquery_sed_log = get_sed_logscale_sim_vec(data_df_log, entropy_dict_log_sed, initial_query_emb_log)

# Salvataggio in session_data

        
    initial_prob_pichunter= compute_first_probability(data_df, initial_query_emb)
    df_display_col_names_log = df_results['imgId'].head(display_number).tolist()
    df_display_log = data_df_log[df_display_col_names_log]
    session_data[session_id]["data_df"] = data_df
    session_data[session_id]["df_display"] = df_display
    session_data[session_id]["data_df_log"] = data_df_log
    session_data[session_id]["df_display_log"] = df_display_log
    session_data[session_id]["query_value_rocchio"] = initial_query_emb
    session_data[session_id]["score_value_polyadic"] = rho0_polyquery_sed_log
    session_data[session_id]["entropy_dict_value"] = entropy_dict_log_sed
    session_data[session_id]["query_value_decap"] = initial_query_emb_log

    session_data[session_id]["new_prob_values_pichunter_star"] = None
    session_data[session_id]["new_prob_values_pic"] = None
    session_data[session_id]["query_value_svm"] = initial_query_emb
    session_data[session_id]["initial_prob_pichunter"] = initial_prob_pichunter
    session_data[session_id]["score_value_polyquery_msed_log"] = rho0_polyquery_msed_log
    session_data[session_id]["entropy_dict_value"] = entropy_dict_log
    session_data[session_id]["query_value_decap"] = initial_query_emb_log
    print('shape of data_df_log:', data_df_log.shape)
    image_urls = ["https://visione.isti.cnr.it/frames/{}/{}.png".format(img_id.split('-')[0], img_id) for img_id in img_ids]
    

    return jsonify({'image_urls': image_urls, 'img_ids': img_ids,'session_id': session_id,})
@app.route('/save_and_update', methods=['POST'])
def save_and_update():
    """Gestisce l'aggiornamento della sessione basato sulle immagini selezionate dall'utente."""

  
    data = request.get_json()
    session_id = data.get('session_id')

    
    if session_id not in sessions:
        return jsonify({'error': 'Session expired or invalid'}), 400

    relevant_image_ids = data.get('relevant_images_ids', [])
    non_relevant_image_ids = data.get('non_relevant_images_ids', [])
    relevant_image_ids_temp = data.get('relevant_images_ids_temp', [])
    non_relevant_image_ids_temp = data.get('non_relevant_images_ids_temp', [])


    selected_algorithm = data.get('selected_algorithm', 'rocchio').lower()


   
    data_df = session_data[session_id]["data_df"]
    
    df_display = session_data[session_id]["df_display"]
    print('shape of df_display:', df_display.shape)
    
    clicked_ids = set(relevant_image_ids_temp + non_relevant_image_ids_temp)


    columns_list = list(df_display.columns)

    
    max_clicked_position = -1 
    for i, col in enumerate(columns_list):
        if col in clicked_ids:
            max_clicked_position = max(max_clicked_position, i)
   
    num_columns = max(100, max_clicked_position + 1)

    
    df_display = df_display[columns_list[:num_columns]]

    with open('relevant_images_ids.json', 'w') as f_relevant:
        json.dump({'relevant_images_ids': relevant_image_ids}, f_relevant)

    with open('non_relevant_images_ids.json', 'w') as f_non_relevant:
        json.dump({'non_relevant_images_ids': non_relevant_image_ids}, f_non_relevant)

    #########################################################################
    #### ROCCHIO
    #########################################################################
    if selected_algorithm == 'rocchio':
        df_display, new_query, time_of_search,  = rocchio_single_step(
            data_df, df_display, relevant_image_ids, non_relevant_image_ids,
            alpha=1.2, beta=2.4, gamma=2.6, fun_name="euclidean",
            initial_query=session_data[session_id]["query_value_rocchio"]
        )
        session_data[session_id]["query_value_rocchio"] = new_query
  
    #########################################################################
    #### PICHUNTER STAR
    ######################################################################### 
    elif selected_algorithm == 'pichunter-star':
        df_display, new_prob_values_star, time_of_search= pichunter_single_step_star(
            data_df, df_display, relevant_image_ids_temp, non_relevant_image_ids_temp,
            fun_name="softmin", initial_prob=session_data[session_id]["initial_prob_pichunter"],
            temperature=82.10553
        )
        session_data[session_id]["initial_prob_pichunter"] = new_prob_values_star
    
    #########################################################################
    #### PICHUNTER
    ######################################################################### 
    elif selected_algorithm == 'pichunter':
        df_display, new_prob_values, time_of_search = pichunter_single_step_star(
            data_df, df_display, relevant_image_ids_temp, [],
            fun_name="softmin", initial_prob=session_data[session_id]["initial_prob_pichunter"],
            temperature=82.10553
        )
        session_data[session_id]["initial_prob_pichunter"] = new_prob_values
    
    #########################################################################
    #### SVM
    ######################################################################### 
    elif selected_algorithm == 'svm':
        df_display, _, time_of_search,  = svm_single_step(
            data_df, df_display, relevant_image_ids, non_relevant_image_ids,query=session_data[session_id]["query_value_svm"],
        )
    
    #########################################################################
    #### POLYADIC SED
    ######################################################################### 
    elif selected_algorithm == 'polyadic-sed':
     
        df_display, new_scores, precomputed_dict_sed, entropy_dict_sed, time_of_search = poly_sed_logscale_single_step(
        session_data[session_id]["data_df_log"], session_data[session_id]["df_display_log"],
        relevant_image_ids, non_relevant_image_ids,
        precomputed_dict_initial=session_data[session_id]["precomputed_dict_polyquery_sed_log_value"],
        alpha=1, beta=1.6, gamma=1.6,
        initial_query=session_data[session_id]["query_value_decap"],  # ✅ embedding log della query
        initial_scores=session_data[session_id]["score_value_polyadic"],  # ✅ rho0
        entropy_dict=session_data[session_id]["entropy_dict_value"]
    )

    
    #########################################################################
    #### POLYADIC MSED
    #########################################################################
    elif selected_algorithm == 'polyadic-msed':
        df_display, new_scores, precomputed_dict, entropy_dict, time_of_search = poly_msed_logscale_single_step(
        session_data[session_id]["data_df_log"], 
        session_data[session_id]["df_display_log"],
        relevant_image_ids, 
        non_relevant_image_ids,
        precomputed_dict_initial=session_data[session_id]["precomputed_dict_polyquery_msed_log_value"],
        alpha=0.4, 
        beta=2, 
        gamma=1.4, 
        initial_query=session_data[session_id]["query_value_decap"],  # ✅ usa la query logaritmica
        initial_scores=session_data[session_id]["score_value_polyquery_msed_log"],  # ✅ usa rho0
        entropy_dict=session_data[session_id]["entropy_dict_value"]
    )


    else:
        print("Errore: Algoritmo non riconosciuto, viene utilizzato Rocchio come fallback.")
        df_display, new_query,time_of_search, = rocchio_single_step(
            data_df, df_display, relevant_image_ids_temp, non_relevant_image_ids_temp,
            alpha=0.75, beta=1, gamma=0.75, fun_name="euclidean",
            initial_query=session_data[session_id]["query_value_rocchio"]
        )
        session_data[session_id]["query_value_rocchio"] = new_query

    
    new_img_ids = df_display.columns.tolist()
    new_image_urls = [f"https://visione.isti.cnr.it/frames/{img_id.split('-')[0]}/{img_id}.png" for img_id in new_img_ids]

    # Log  debug
    print(f"Non relevant image ids: {non_relevant_image_ids}")
    print(f"Relevant image ids: {relevant_image_ids}")
    print(f"Non relevant image ids temp: {non_relevant_image_ids_temp}")
    print(f"Relevant image ids temp: {relevant_image_ids_temp}")
    print(f'Time of search: {time_of_search}')
    print('shape of df_display:', df_display.shape)
   
    response_data = {
        'status': 'success',
        'session_id': session_id,
        'relevant_images_ids': relevant_image_ids,
        'non_relevant_images_ids': non_relevant_image_ids,
        'relevant_images_ids_temp': relevant_image_ids_temp,
        'non_relevant_images_ids_temp': non_relevant_image_ids_temp,
        'selected_algorithm': selected_algorithm,
        'new_image_ids': new_img_ids,
        'new_image_urls': new_image_urls,
        'total_time': f"{time_of_search.total_seconds()} seconds"
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

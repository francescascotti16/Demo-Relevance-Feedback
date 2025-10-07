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
    return send_from_directory('static', 'demo_copy.html')

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
    
    query = json.dumps({"query": [{"textual": query_orig}], "parameters": [{"textualMode": textual_mode, "occur": "and", "simReorder": "false"}]})
   
    

    results = requests.post(host + '/services/core/search', data={'query': query, 'sortbyvideo': False, 'maxres': max_rank},  verify=False)
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
@app.route('/save_and_update', methods=['POST'])
def save_and_update():
    """Gestisce l'aggiornamento della sessione basato sulle immagini selezionate dall'utente."""
    data = request.get_json()
    session_id = data.get('session_id')

    if session_id not in sessions:
        return jsonify({'error': 'Session expired or invalid'}), 400

    # input feedback
    relevant_image_ids         = data.get('relevant_images_ids', [])
    non_relevant_image_ids     = data.get('non_relevant_images_ids', [])
    relevant_image_ids_temp    = data.get('relevant_images_ids_temp', [])
    non_relevant_image_ids_temp= data.get('non_relevant_images_ids_temp', [])

    selected_algorithm = data.get('selected_algorithm', 'rocchio').lower()

    # dati di sessione
    data_df    = session_data[session_id]["data_df"]
    df_display = session_data[session_id]["df_display"]

    # shrink display in base a quanto cliccato
    clicked_ids = set(relevant_image_ids_temp + non_relevant_image_ids_temp)
    cols = list(df_display.columns)
    max_pos = max([i for i,c in enumerate(cols) if c in clicked_ids] + [-1])
    df_display = df_display[cols[: max(100, max_pos+1) ]]

    # salva log
    with open('relevant_images_ids.json', 'w') as f: json.dump({'relevant_images_ids': relevant_image_ids}, f)
    with open('non_relevant_images_ids.json', 'w') as f: json.dump({'non_relevant_images_ids': non_relevant_image_ids}, f)

    # PREPARO response_data di base
    response_data = {
        'status': 'success',
        'session_id': session_id,
        'relevant_images_ids': relevant_image_ids,
        'non_relevant_images_ids': non_relevant_image_ids,
        'relevant_images_ids_temp': relevant_image_ids_temp,
        'non_relevant_images_ids_temp': non_relevant_image_ids_temp,
        'selected_algorithm': selected_algorithm,
    }

    # ======================
    # SWITCH sui metodi
    # ======================
    if selected_algorithm == 'rocchio':
        print()
        print("Relevant image IDs:", relevant_image_ids)
        print()
        df_display, new_query, time_of_search = rocchio_single_step(
            data_df, df_display,
            relevant_image_ids, non_relevant_image_ids,
            alpha=1.2, beta=2.4, gamma=2.6,
            fun_name="euclidean",
            initial_query=session_data[session_id]["query_value_rocchio"]
        )
        session_data[session_id]["query_value_rocchio"] = new_query

        # aggiungo risultati singolo metodo
        new_ids  = df_display.columns.tolist()
        new_urls = [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in new_ids]
        response_data.update({
            'new_image_ids':  new_ids,
            'new_image_urls': new_urls,
            'total_time':     f"{time_of_search.total_seconds()} seconds"
        })

    elif selected_algorithm == 'pichunter-star':
        df_display, new_probs, time_of_search = pichunter_single_step_star(
            data_df, df_display,
            relevant_image_ids_temp, non_relevant_image_ids_temp,
            fun_name="softmin",
            initial_prob=session_data[session_id]["initial_prob_pichunter"],
            temperature=82.10553
        )
        session_data[session_id]["initial_prob_pichunter"] = new_probs

        new_ids  = df_display.columns.tolist()
        new_urls = [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in new_ids]
        response_data.update({
            'new_image_ids':  new_ids,
            'new_image_urls': new_urls,
            'total_time':     f"{time_of_search.total_seconds()} seconds"
        })

    elif selected_algorithm == 'pichunter':
        df_display, new_probs, time_of_search = pichunter_single_step_star(
            data_df, df_display,
            relevant_image_ids_temp, [],
            fun_name="softmin",
            initial_prob=session_data[session_id]["initial_prob_pichunter"],
            temperature=82.10553
        )
        session_data[session_id]["initial_prob_pichunter"] = new_probs

        new_ids  = df_display.columns.tolist()
        new_urls = [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in new_ids]
        response_data.update({
            'new_image_ids':  new_ids,
            'new_image_urls': new_urls,
            'total_time':     f"{time_of_search.total_seconds()} seconds"
        })

    elif selected_algorithm == 'svm':
        df_display, _, time_of_search = svm_single_step(
            data_df, df_display,
            relevant_image_ids, non_relevant_image_ids,
            query=session_data[session_id]["query_value_svm"]
        )
        new_ids  = df_display.columns.tolist()
        new_urls = [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in new_ids]
        response_data.update({
            'new_image_ids':  new_ids,
            'new_image_urls': new_urls,
            'total_time':     f"{time_of_search.total_seconds()} seconds"
        })

    elif selected_algorithm == 'polyadic-sed':
        df_display, _, _, _, time_of_search = poly_sed_logscale_single_step(
            session_data[session_id]["data_df_log"],
            session_data[session_id]["df_display_log"],
            relevant_image_ids, non_relevant_image_ids,
            precomputed_dict_initial=session_data[session_id]["precomputed_dict_polyquery_sed_log_value"],
            alpha=1, beta=1.6, gamma=1.6,
            initial_query=session_data[session_id]["query_value_decap"],
            initial_scores=session_data[session_id]["score_value_polyadic"],
            entropy_dict=session_data[session_id]["entropy_dict_value"]
        )
        new_ids  = df_display.columns.tolist()
        new_urls = [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in new_ids]
        response_data.update({
            'new_image_ids':  new_ids,
            'new_image_urls': new_urls,
            'total_time':     f"{time_of_search.total_seconds()} seconds"
        })

    elif selected_algorithm == 'polyadic-msed':
        df_display, _, _, _, time_of_search = poly_msed_logscale_single_step(
            session_data[session_id]["data_df_log"],
            session_data[session_id]["df_display_log"],
            relevant_image_ids, non_relevant_image_ids,
            precomputed_dict_initial=session_data[session_id]["precomputed_dict_polyquery_msed_log_value"],
            alpha=0.4, beta=2, gamma=1.4,
            initial_query=session_data[session_id]["query_value_decap"],
            initial_scores=session_data[session_id]["score_value_polyquery_msed_log"],
            entropy_dict=session_data[session_id]["entropy_dict_value"]
        )
        new_ids  = df_display.columns.tolist()
        new_urls = [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in new_ids]
        response_data.update({
            'new_image_ids':  new_ids,
            'new_image_urls': new_urls,
            'total_time':     f"{time_of_search.total_seconds()} seconds"
        })


    else:
        # fallback su Rocchio
        df_display, new_query, time_of_search = rocchio_single_step(
            data_df, df_display,
            relevant_image_ids_temp, non_relevant_image_ids_temp,
            alpha=0.75, beta=1, gamma=0.75,
            fun_name="euclidean",
            initial_query=session_data[session_id]["query_value_rocchio"]
        )
        session_data[session_id]["query_value_rocchio"] = new_query
        new_ids  = df_display.columns.tolist()
        new_urls = [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in new_ids]
        response_data.update({
            'new_image_ids':  new_ids,
            'new_image_urls': new_urls,
            'total_time':     f"{time_of_search.total_seconds()} seconds"
        })

    # restituisco sempre lo stesso oggetto JSON
    return jsonify(response_data)
from copy import deepcopy

@app.route('/compare_all_methods', methods=['POST'])
def compare_all_methods():
    data = request.get_json()
    sid = data.get('session_id')
    if sid not in sessions:
        return jsonify({'error':'Session invalid'}), 400

    rel = data.get('relevant_images_ids', [])
    nonrel = data.get('non_relevant_images_ids', [])

    # session data
    data_df    = session_data[sid]["data_df"]
    df_disp0   = session_data[sid]["df_display"]
    data_df_log   = session_data[sid]["data_df_log"]
    df_disp_log   = session_data[sid]["df_display_log"]

    comparisons = {}

    only_negatives = (len(rel) == 0 and len(nonrel) > 0)

    # 1) Rocchio – sempre ok
    df_tmp = deepcopy(df_disp0)
    df_new, new_q, t = rocchio_single_step(
        data_df, df_tmp, rel, nonrel,
        alpha=1.2, beta=2.4, gamma=2.6,
        fun_name="euclidean",
        initial_query=session_data[sid]["query_value_rocchio"]
    )
    comparisons['rocchio'] = {
      'ids':  df_new.columns.tolist(),
      'urls': [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in df_new.columns],
      'time': f"{t.total_seconds():.2f}s"
    }

    # 2) PicHunter – SKIP se only_negatives (richiede positivi)
    if not only_negatives:
        df_tmp = deepcopy(df_disp0)
        df_new, new_p, t = pichunter_single_step_star(
            data_df, df_tmp, rel, [],  # PicHunter "base": solo positivi
            fun_name="softmin",
            initial_prob=session_data[sid]["initial_prob_pichunter"],
            temperature=82.10553
        )
        comparisons['pichunter'] = {
          'ids':  df_new.columns.tolist(),
          'urls': [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in df_new.columns],
          'time': f"{t.total_seconds():.2f}s"
        }

    # 3) PicHunter-star – OK anche con soli negativi (usa rel+nonrel)
    df_tmp = deepcopy(df_disp0)
    df_new, new_ps, t = pichunter_single_step_star(
        data_df, df_tmp, rel, nonrel,
        fun_name="softmin",
        initial_prob=session_data[sid]["initial_prob_pichunter"],
        temperature=82.10553
    )
    comparisons['pichunter-star'] = {
      'ids':  df_new.columns.tolist(),
      'urls': [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in df_new.columns],
      'time': f"{t.total_seconds():.2f}s"
    }

    # 4) SVM – SKIP se only_negatives (in genere richiede almeno un positivo)
    if not only_negatives:
        df_tmp = deepcopy(df_disp0)
        df_new, _, t = svm_single_step(
            data_df, df_tmp, rel, nonrel,
            query=session_data[sid]["query_value_svm"]
        )
        comparisons['svm'] = {
          'ids':  df_new.columns.tolist(),
          'urls': [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in df_new.columns],
          'time': f"{t.total_seconds():.2f}s"
        }

    # 5) Polyadic-SED – sempre ok
    df_tmp_log = deepcopy(df_disp_log)
    df_new, _, _, _, t = poly_sed_logscale_single_step(
        data_df_log, df_tmp_log, rel, nonrel,
        precomputed_dict_initial=session_data[sid]["precomputed_dict_polyquery_sed_log_value"],
        alpha=1, beta=1.6, gamma=1.6,
        initial_query=session_data[sid]["query_value_decap"],
        initial_scores=session_data[sid]["score_value_polyadic"],
        entropy_dict=session_data[sid]["entropy_dict_value"]
    )
    comparisons['polyadic-sed'] = {
      'ids': df_new.columns.tolist(),
      'urls': [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in df_new.columns],
      'time': f"{t.total_seconds():.2f}s"
    }

    # 6) Polyadic-MSED – sempre ok
    df_tmp_log = deepcopy(df_disp_log)
    df_new, _, _, _, t = poly_msed_logscale_single_step(
        data_df_log, df_tmp_log, rel, nonrel,
        precomputed_dict_initial=session_data[sid]["precomputed_dict_polyquery_msed_log_value"],
        alpha=0.4, beta=2, gamma=1.4,
        initial_query=session_data[sid]["query_value_decap"],
        initial_scores=session_data[sid]["score_value_polyquery_msed_log"],
        entropy_dict=session_data[sid]["entropy_dict_value"]
    )
    comparisons['polyadic-msed'] = {
      'ids': df_new.columns.tolist(),
      'urls': [f"https://visione.isti.cnr.it/frames/{i.split('-')[0]}/{i}.png" for i in df_new.columns],
      'time': f"{t.total_seconds():.2f}s"
    }

    return jsonify({'status':'success','comparisons':comparisons})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

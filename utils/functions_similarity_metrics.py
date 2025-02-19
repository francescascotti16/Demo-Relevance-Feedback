from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import additive_chi2_kernel
from scipy.spatial.distance import jensenshannon
import numpy as np
from scipy.special import softmax as sp_softmax
from datetime import datetime

def softmax(x,temperature=1, axis=1, keepdims=True):
    '''
    Parameters:
    x: input array
    temperature: temperature parameter
    Returns:
    softmax: softmax of the input array
    '''    
    
    e_x = np.exp(x/temperature)
    return e_x / e_x.sum(axis=axis, keepdims=keepdims)

def logistic(x,temperature=1):
    '''
    Parameters:
    x: input array
    temperature: temperature parameter
    Returns:
    num: logistic of the input array
    '''
    num=1/(1+np.exp(-x/temperature))
    return num/num.sum()

def complexity(x):
    start_time = datetime.now()
    '''
    Parameters:
    x: input probabilistic array 
    Returns:
    complexity: complexity of the input array
    '''
 
    shannon_entropy_, time_shannon_entropy = shannon_entropy(x)
    result =np.exp(shannon_entropy_)
    end_time = datetime.now()
    total_time = end_time - start_time
    return result, total_time, time_shannon_entropy

def shannon_entropy(x):
    start_time = datetime.now()
    '''
    Parameters:
    x: input probabilistic array 
    Returns:
    shannon_entropy: shared entropy of the input array
    '''
    # find if there are zero or negative values in the input array
    if np.any(x <= 0):
        raise ValueError("Invalid values encountered in input array x")
    y= np.where(x == 0, 1, x)  # Replace zero or negative values with epsilon
    ##shall we do #epsilon = 1e-10  # Small epsilon to avoid zero or negative values y= np.where(x < epsilon, 1, x)  # x = np.where(x <= 0, epsilon, x)  # Replace zero or negative values with epsilon
    if np.any(y <= 0):
        raise ValueError("Invalid values encountered in input array x")
    log_x = np.log(y)
    shannon_entropy = -np.sum(x * log_x)
 
    end_time = datetime.now()
    total_time = end_time - start_time
    return shannon_entropy, total_time


def sed(x, y):
    start_time = datetime.now()
    '''
    Parameters:
    x: input probabilistic array 
    y: input probabilistic array 
    Returns:
    sed: symmetric entropy distance between the two input arrays
    '''
    # Compute the average of the two arrays
    avg = (x + y) / 2
    # Compute the complexity of the average array
    complexity_avg, complexity_time_avg, time_shannon_entropy = complexity(avg)
  
    # Compute the product of the complexity of the two arrays
    complexity_x, complexity_time_x, time_shannon_entropy = complexity(x)
    complexity_y, complexity_time_y , time_shannon_entropy= complexity(y)
    
    complexity_product = (complexity_x * complexity_y) ** (1 / 2)
    # Compute the symmetric entropy distance
    sed = complexity_avg / complexity_product
    end_time = datetime.now()
    total_time = end_time - start_time
    return sed-1, total_time, complexity_time_avg, complexity_time_x, complexity_time_y


# def triangular(x, y):
#     xy_sum = x + y
#     xy_sum[xy_sum == 0] = 1  # Evita la divisione per zero
#     res = (x * y) / xy_sum
#     return 1 - 2*np.sum(res)

# def triangular_matrix_matrix(x_df, y_df):
#     x_matrix = x_df.values
#     y_matrix = y_df.values

#     results = np.zeros((x_matrix.shape[0], y_matrix.shape[0]))
    
#     for i in range(x_matrix.shape[0]):
#         for j in range(y_matrix.shape[0]):
#             results[i, j] = triangular(x_matrix[i], y_matrix[j]
            
#     return results

def jensen_shannon_matrix(x_df, y_df):
    if isinstance(x_df, np.ndarray):
        x_matrix = x_df
    else:
        x_matrix = x_df.values

    if isinstance(y_df, np.ndarray):
        y_matrix = y_df
    else:
        y_matrix = y_df.values

    results = np.zeros((x_matrix.shape[0], y_matrix.shape[0]))
    
    for i in range(x_matrix.shape[0]):
        for j in range(y_matrix.shape[0]):
            results[i, j] = jensenshannon(x_matrix[i], y_matrix[j])
            
    return results

def sed_matrix(x_df, y_df):
    print('x_df',x_df.shape)
    print('y_df',y_df.shape)
    if isinstance(x_df, np.ndarray):
        x_matrix = x_df
    else:
        x_matrix = x_df.values

    if isinstance(y_df, np.ndarray):
        y_matrix = y_df
    else:
        y_matrix = y_df.values

    results = np.zeros((x_matrix.shape[0], y_matrix.shape[0]))
    
    for i in range(x_matrix.shape[0]):
        for j in range(y_matrix.shape[0]):
            sedd, total_time_sed, complexity_time_avg, complexity_time_x, complexity_time_y=sed(x_matrix[i], y_matrix[j])
            results[i, j] =sedd
            
    return results, total_time_sed, complexity_time_avg, complexity_time_x, complexity_time_y
def compute_similarity_score(query,dataset_df, fun_name="cosine") :
    start_time = datetime.now()
    '''
    query: query vector
    dataset_df: DataFrame with the dataset
    fun_name: function to calculate the similarity or metric model
    Returns:
    score: score of the query
    '''
    
    # Function to calculate the score of a query
    if fun_name == "dotproduct":
        score = np.dot(dataset_df.T, query) #.values 
    elif fun_name == "cosine":
        score = cosine_similarity(dataset_df.T, query.T)
    elif fun_name == "jsd":
        score= 1- jensen_shannon_matrix (dataset_df.T, query.T)
    elif fun_name == "triangular":
        score=1+0.5*additive_chi2_kernel(dataset_df.T, query.T) #1-triangular_matrix_matrix(dataset_df.T, query.T)    
    elif fun_name== "sed":
        sed_matrixx, total_time_sed, complexity_time_avg, complexity_time_x, complexity_time_y=sed_matrix(dataset_df.T, query.T)
        score= 1-sed_matrixx
        

    else: 
        print(f"ERR The function {fun_name} is not implemented. Using cosine similarity.")  
        score = cosine_similarity(dataset_df.T, query.T)
    end_time = datetime.now()
    total_time = end_time - start_time
    return score, total_time_sed, complexity_time_avg, complexity_time_x, complexity_time_y, total_time

def get_similarity_matrix(data1_df, dataset_df,fun_name="dotproduct"): 
    '''
    data1_df: DataFrame with the queries or display
    dataset_df: DataFrame with the dataset
    similarity_fun_name: similatity function to calculate 

    '''
    # Function to get the similarity matrix between the display and the whole dataset
    n = data1_df.shape[1]
    m = dataset_df.shape[1]
    similarity = np.zeros((m, n))
    #switch case to select the function to calculate the user model
    if  fun_name == "dotproduct":
        similarity=1+ np.dot(dataset_df.T, data1_df) #.values 
    elif fun_name == "cosine":
        similarity = 1 + cosine_similarity(dataset_df.T, data1_df.T)
    elif fun_name == "jsd":
        similarity= 1-jensen_shannon_matrix(dataset_df.T, data1_df.T) 
    elif fun_name == "triangular":
        similarity= 1+0.5*additive_chi2_kernel(dataset_df.T, data1_df.T) # 1-triangular_matrix_matrix(dataset_df.T, data1_df.T)
    elif fun_name== "sed":
        similarity= 1-sed_matrix(dataset_df.T, data1_df.T)
    else: 
         print(f"ERR The function {fun_name} is not implemented. Using cosine similarity.")  
         similarity = 1+ cosine_similarity(dataset_df.T, data1_df.T)
 
    return similarity


def get_similarity_matrix_temperature(data1_df, dataset_df,user_model_fun_name="softmin", temperature=1): #only for picHunter
    '''
    data1_df: DataFrame with the display
    dataset_df: DataFrame with the dataset
    user_model_fun_name: function to calculate the user model

    '''
    start_time = datetime.now()
    # Function to get the similarity matrix between the display and the whole dataset
    n = data1_df.shape[1]
    m = dataset_df.shape[1]
    similarity = np.zeros((m, n))
    #switch case to select the function to calculate the user model

    if user_model_fun_name == "softmin":
        similarity = np.exp(-euclidean_distances(dataset_df.T, data1_df.T)/temperature)
    elif user_model_fun_name == "softmax_cosine":
        similarity = np.exp(cosine_similarity(dataset_df.T, data1_df.T)/temperature)
    elif user_model_fun_name == "l1_normalized_cosine":
        similarity =(1.001+cosine_similarity(dataset_df.T, data1_df.T))/temperature  
    else: 
        print(f"ERR The user model function {user_model_fun_name} is not implemented. Using softmin similarity.")  
        similarity = np.exp(-euclidean_distances(dataset_df.T, data1_df.T)/temperature)
    end_time = datetime.now()
    total_time=end_time-start_time
    return similarity , total_time
def get_fast_soft_similarity_matrix(n_display,selected_df, dataset_df,avg_norm,user_model_fun_name="softmin"): 
    '''
    selected_df: DataFrame with the image from display selected by the user
    dataset_df: DataFrame with the dataset
    avg_norm: temperature parameter for the softmax/min function. We assume it is average of a vector of n_display dist/sim values
    user_model_fun_name: function to calculate the user model
    '''
    # Function to get the similarity matrix between the display and the whole dataset
    n = selected_df.shape[1]
    n_to_simulate=n_display-n
    m = dataset_df.shape[1]
    similarity = np.zeros((m, n))
   
 
    if user_model_fun_name == "softmin":
        #the original PicHunter softmin function  with Euclidean distance
        similarity = np.exp(-euclidean_distances(dataset_df.T, selected_df.T)/avg_norm)
        mean_soft_sim_value= np.exp(-1/np.sqrt(n_display))
    elif user_model_fun_name == "softmax_cosine":
        #the original PicHunter softmin function  with Euclidean distance
        similarity = np.exp(cosine_similarity(dataset_df.T, selected_df.T)/avg_norm)
        mean_soft_sim_value= np.exp(1/np.sqrt(n_display))
    elif user_model_fun_name == "l1_normalized_cosine":
        similarity = 1 + cosine_similarity(dataset_df.T, selected_df.T) #cosine similarity between the columns of dataset_df and data1_df (transposed versions of these DataFrames).
        mean_soft_sim_value=avg_norm/np.sqrt(n_display)
    else: 
        print(f"ERR The user model function {user_model_fun_name} is not implemented. Using softmin similarity.")  
        similarity = np.exp(-euclidean_distances(dataset_df.T, selected_df.T)/avg_norm)
        mean_soft_sim_value= np.exp(-1/np.sqrt(n_display))
      
  
    sum_similarities=similarity.sum(axis=1)[:, np.newaxis] + mean_soft_sim_value*n_to_simulate
    return similarity/sum_similarities #ow-wise normalized version of the similarity scores between the columns of dataset_df and data1_df


def get_soft_similarity_matrix(data1_df, dataset_df,user_model_fun_name="softmin", temperature=1): 
    '''
    data1_df: DataFrame with the display
    dataset_df: DataFrame with the dataset
    user_model_fun_name: function to calculate the user model

    '''
    # Function to get the similarity matrix between the display and the whole dataset
    n = data1_df.shape[1]
    m = dataset_df.shape[1]
    similarity = np.zeros((m, n))
    #switch case to select the function to calculate the user model

    if user_model_fun_name == "softmin":
        #the original PicHunter softmin function  with Euclidean distance
        similarity = np.exp(-euclidean_distances(dataset_df.T, data1_df.T)/temperature)
    elif user_model_fun_name == "softmax_cosine":
        #the original PicHunter softmin function  with Euclidean distance
        similarity = np.exp(cosine_similarity(dataset_df.T, data1_df.T)/temperature)
    elif user_model_fun_name == "l1_normalized_cosine":
        similarity = 1 + cosine_similarity(dataset_df.T, data1_df.T)/temperature #cosine similarity between the columns of dataset_df and data1_df (transposed versions of these DataFrames).
    else: 
        print(f"ERR The user model function {user_model_fun_name} is not implemented. Using softmin similarity.")  
        similarity = np.exp(-euclidean_distances(dataset_df.T, data1_df.T)/temperature)
 
    return similarity/similarity.sum(axis=1)[:, np.newaxis] #ow-wise normalized version of the similarity scores between the columns of dataset_df and data1_df






def get_distance_matrix(data1_df, dataset_df, fun_name="euclidean"):
     
    '''
    data1_df: DataFrame with the data queries or all display
    dataset_df: DataFrame with the dataset
    fun_name: similatity function to calculate 

    '''
    start_time = datetime.now()
    # Function to get the similarity matrix between the display and the whole dataset
    n = data1_df.shape[1]
    m = dataset_df.shape[1]
    distance_matrix = np.zeros((m, n))
    #switch case to select the function to calculate the user model
    if  fun_name == "euclidean":
        distance_matrix=euclidean_distances(dataset_df.T, data1_df.T)
    elif fun_name == "triangular":
        distance_matrix= -0.5*additive_chi2_kernel(dataset_df.T, data1_df.T) 
    elif fun_name == "jsd":
        distance_matrix= jensen_shannon_matrix(dataset_df.T, data1_df.T)
    elif fun_name == "sed":
        distance_matrix= sed_matrix(dataset_df.T, data1_df.T)
    else: 
         print(f"ERR The function {fun_name} is not implemented. Using Euclidean distance.")  
         distance_matrix=euclidean_distances(dataset_df.T, data1_df.T)
    end_time = datetime.now()
    total_time=end_time-start_time
    return distance_matrix , total_time



# def precomputed_sum_product(df_with_complexity,action, 
#                             precomputed_sum_pos=0,
#                             precomputed_sum_neg=0,
#                             precomputed_product_complexity_neg=1,
#                             precomputed_product_complexity_pos=1):

#     '''
#     Parameters:
#     df_with_complexity: DataFrame with the dataset and the complexity of each image, image_id are the index. 
#         The dataframe has two columns 'data','complexity'. Data column stores a numpy array for each image. Complexity column stores the complexity of the image (number)  
#         action: list of tuple containing the selected images and their relevance (img_id, relevance)
#         precomputed_sum_pos: precomputed sum of the positive images + data
#         precomputed_sum_neg: precomputed sum of the negative images +data
#         precomputed_product_complexity_neg: precomputed product of the complexity of the negative images and the data complexity 
#         precomputed_product_complexity_pos: precomputed product of the complexity of the positive images and the data complexity 
#     '''
    
#     relevant_ids=[im for im, rel in action if rel == 1]
#     non_relevant_ids=[im for im, rel in action if rel == 0]

#     for el in non_relevant_ids: 
#         precomputed_sum_neg+=df_with_complexity.loc[el,'data']
#         precomputed_product_complexity_neg*=(df_with_complexity.loc[el,'complexity'])
        

#     for el in relevant_ids : 
#         precomputed_sum_pos+=df_with_complexity.loc[el,'data']
#         precomputed_product_complexity_pos*=(df_with_complexity.loc[el,'complexity'])
        
#     return  precomputed_sum_pos, precomputed_sum_neg, precomputed_product_complexity_pos, precomputed_product_complexity_neg



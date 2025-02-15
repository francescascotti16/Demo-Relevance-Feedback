
from datetime import datetime


def create_display(data_df, score, n_display, is_ascending=False):
    
    '''
    Parameters:
    data_df: DataFrame with the dataset one column for each image and the last row with the GT relevance
    score: list with the score of each image
    n_display: int, number of images in the display
    is_ascending: boolean, to sort the display in ascending order
    
    Returns:
    display_df: DataFrame with the display
    '''
    start_time = datetime.now()
    # Function to create and transpose the display dataset
    df_copy_t = data_df.copy().transpose()   
    df_copy_t['score'] = score
    df_sorted = df_copy_t.sort_values(by='score', ascending=is_ascending)
    display_df_t = df_sorted.head(n_display)
    display_df_t = display_df_t.copy()
    display_df_t.drop(columns=['score'], errors='ignore', inplace=True)
    display_df = display_df_t.transpose()
    end_time = datetime.now()
    total_time=end_time-start_time
    return display_df, total_time

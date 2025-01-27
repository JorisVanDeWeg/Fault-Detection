from PU_functions import *
from CRWU_functions import *
import pandas as pd
import time 
window_duration = 2
overlap = 0

# Generate 
PU = True
CRWU = False

if PU:
    signal_indices = [6]

    start_time = time.time()
    pu_base_dir_list = ['C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/PU/Healthy',
                        'C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/PU/Real Damages']
                        # 'C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/PU/Real Damages without double']
                        # 'C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/PU/Artificially Damaged']

    pu_features_df, pu_time_series_df = load_pu_dataset(pu_base_dir_list, 
                                                    window_duration=window_duration, 
                                                    overlap=overlap, 
                                                    signal_indices=signal_indices, 
                                                    classification_type="multiclass")
    print(f"Elapsed time for constructing dataframes: {time.time() - start_time:.6f} seconds")  # Print elapsed time


    ## Time series SAVE
    start_time = time.time()
    pu_time_series_df.to_parquet('Dataframes/pu_time_series_df.parquet', engine='pyarrow')
    print(f"Elapsed time for SAVING time series: {time.time() - start_time:.6f} seconds")  # Print elapsed time

    ## Features SAVE (process is already fast enough)
    pu_features_df.to_csv('Dataframes/pu_features_df.csv', index=False)

    ## Time series LOAD (tester)
    # start_time = time.time()
    # loaded_df = pd.read_parquet('Dataframes/pu_time_series_df.parquet', engine='pyarrow')
    # print(f"Elapsed time for LOADING time series: {time.time() - start_time:.6f} seconds")  # Print elapsed time


if CRWU:
    crwu_base_dir_list = ['C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/CWRU/Data/12k_DE',
                        'C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/CWRU/Data/Normal']

    crwu_features_df, crwu_time_series_df = load_crwu_dataset(crwu_base_dir_list, 
                                                            window_duration=window_duration,
                                                            overlap=overlap,
                                                            classification_type="multiclass")
    
    crwu_features_df.to_csv('Dataframes/crwu_features_df.csv', index=False)
    crwu_time_series_df.to_csv('Dataframes/crwu_time_series_df.csv', index=False)
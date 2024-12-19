from PU_functions import *
from CRWU_functions import *

window_duration = 4
overlap = 0

# Generate 
PU = True
CRWU = True

if PU:
    signal_indices = [6]

    pu_base_dir_list = ['C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/PU/Healthy',
                        #'C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/PU/Real Damages',]
                        'C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/PU/Real Damages without double']
                        #'C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/PU/Artificially Damaged'

    pu_features_df, pu_time_series_df = load_pu_dataset(pu_base_dir_list, 
                                                    window_duration=window_duration, 
                                                    overlap=overlap, 
                                                    signal_indices=signal_indices, 
                                                    classification_type="multiclass")

    pu_features_df.to_csv('Dataframes (csv)/pu_features_df.csv', index=False)
    pu_time_series_df.to_csv('Dataframes (csv)/pu_time_series_df.csv', index=False)

if CRWU:
    crwu_base_dir_list = ['C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/CWRU/Data/12k_DE',
                        'C:/Users/joris/Documents/EE Master/Year 2/Thesis/Dataset/CWRU/Data/Normal']

    crwu_features_df, crwu_time_series_df = load_crwu_dataset(crwu_base_dir_list, 
                                                            window_duration=window_duration,
                                                            overlap=0,
                                                            classification_type="multiclass")
    
    crwu_features_df.to_csv('Dataframes (csv)/crwu_features_df.csv', index=False)
    crwu_time_series_df.to_csv('Dataframes (csv)/crwu_time_series_df.csv', index=False)
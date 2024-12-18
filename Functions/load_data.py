from PU_functions import *

window_duration = 2
overlap = 0
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
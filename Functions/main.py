import pandas as pd
from general_functions import *
from plot_functions import *

pu_features_df = pd.read_csv('Dataframes (csv)/pu_features_df.csv')
pu_time_series_df = pd.read_csv('Dataframes (csv)/pu_time_series_df.csv')

final_results_pu_r = simple_models(pu_features_df, k=3, split_type='random')
final_results_pu_r.to_csv('Results/final_results_pu_r.csv', index=False)

final_results_pu_sg = simple_models(pu_features_df, k=3, split_type='stratified_group_kfold')
final_results_pu_sg.to_csv('Results/final_results_pu_sg.csv', index=False)

file_name = 'PU_2_seconds'

plot_combined_results2([final_results_pu_r, final_results_pu_sg], ['random', 'grouped kfold'], file_name)
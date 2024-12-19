import pandas as pd
from general_functions import *
from plot_functions import *

# Train ... dataset 
train_PU = True
train_CRWU = True

if train_PU:
    pu_features_df = pd.read_csv('Dataframes (csv)/pu_features_df.csv')
    pu_time_series_df = pd.read_csv('Dataframes (csv)/pu_time_series_df.csv')

    final_results_pu_r = simple_models(pu_features_df, k=3, split_type='random')
    final_results_pu_r.to_csv('Results/final_results_pu_r.csv', index=False)

    final_results_pu_sg = simple_models(pu_features_df, k=3, split_type='stratified_group_kfold')
    final_results_pu_sg.to_csv('Results/final_results_pu_sg.csv', index=False)

    file_name = 'PU_4_seconds'

    plot_combined_results2([final_results_pu_r, final_results_pu_sg], ['random', 'grouped kfold'], file_name)

if train_CRWU:
    crwu_features_df = pd.read_csv('Dataframes (csv)/crwu_features_df.csv')
    crwu_time_series_df = pd.read_csv('Dataframes (csv)/crwu_time_series_df.csv')

    final_results_crwu_r = simple_models(crwu_features_df, k=3, split_type='random')
    final_results_crwu_r.to_csv('Results/final_results_crwu_r.csv', index=False)

    final_results_crwu_sg = simple_models(crwu_features_df, k=3, split_type='stratified_group_kfold')
    final_results_crwu_sg.to_csv('Results/final_results_crwu_sg.csv', index=False)

    file_name = 'CRWU_4_seconds'

    plot_combined_results2([final_results_crwu_r, final_results_crwu_sg], ['random', 'grouped kfold'], file_name)

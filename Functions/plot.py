import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pandas as pd
from plot_functions import *

file_name = 'PU_2_seconds'

final_results_pu_r = pd.read_csv('Results/final_results_pu_r.csv')
final_results_pu_sg = pd.read_csv('Results/final_results_pu_sg.csv')

plot_combined_results2([final_results_pu_r, final_results_pu_sg], ['random', 'grouped kfold'], file_name)
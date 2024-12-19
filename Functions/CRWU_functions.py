import os
import pandas as pd

from scipy.io import loadmat

from general_functions import *


def deduce_labels_crwu(folder_name, recording_id, classification_type="binary"):
    """
    Deduce labels for the CRWU 12k Drive End dataset based on the file name.

    Parameters:
    - folder_name : The name of the folder "normal"
    - bearing_id (str): The name of the .mat file (for example 'IR007_0.mat').
    - classification_type (str): The type of classification ('binary', 'multiclass').

    Returns:
    - int: The label for the given file name.
    """
    # Fault mapping for 12k Drive End
    fault_mapping = {
        # Inner Race Faults
        "IR007": {"fault_type": "Inner Race", "fault_diameter": 0.007},
        "IR014": {"fault_type": "Inner Race", "fault_diameter": 0.014},
        "IR021": {"fault_type": "Inner Race", "fault_diameter": 0.021},
        "IR028": {"fault_type": "Inner Race", "fault_diameter": 0.028},

        # Ball Faults
        "B007": {"fault_type": "Ball", "fault_diameter": 0.007},
        "B014": {"fault_type": "Ball", "fault_diameter": 0.014},
        "B021": {"fault_type": "Ball", "fault_diameter": 0.021},
        "B028": {"fault_type": "Ball", "fault_diameter": 0.028},

        # Outer Race Faults (with positions)
        "OR007@6": {"fault_type": "Outer Race", "fault_position": "Centered (6:00)", "fault_diameter": 0.007},
        "OR014@6": {"fault_type": "Outer Race", "fault_position": "Centered (6:00)", "fault_diameter": 0.014},
        "OR021@6": {"fault_type": "Outer Race", "fault_position": "Centered (6:00)", "fault_diameter": 0.021},
        "OR028@6": {"fault_type": "Outer Race", "fault_position": "Centered (6:00)", "fault_diameter": 0.028},

        "OR007@3": {"fault_type": "Outer Race", "fault_position": "Orthogonal (3:00)", "fault_diameter": 0.007},
        "OR021@3": {"fault_type": "Outer Race", "fault_position": "Orthogonal (3:00)", "fault_diameter": 0.021},

        "OR007@12": {"fault_type": "Outer Race", "fault_position": "Opposite (12:00)", "fault_diameter": 0.007},
        "OR021@12": {"fault_type": "Outer Race", "fault_position": "Opposite (12:00)", "fault_diameter": 0.021},
    }


    bearing_id = recording_id.split('_')[0]

    # Determine if it's Normal data
    if "Normal" in folder_name:
        if classification_type == "binary":
            return 0
        elif classification_type == "multiclass":
            return 0

    if bearing_id in fault_mapping:
        if classification_type == "binary":
            return 1  # Faulty
        elif classification_type == "multiclass":
            fault_type = fault_mapping[bearing_id]["fault_type"]
            if fault_type == "Inner Race":
                return 1
            elif fault_type == "Ball":
                return 2
            elif fault_type == "Outer Race":
                return 3

    raise ValueError(f"File name {folder_name} does not match any known labels.")

def load_crwu_dataset(base_dir_list, window_duration, overlap, fs=12000, classification_type="binary"):
    """
    Load the Case Western Reserve University (CRWU) dataset, process time series into overlapping windows,
    and extract features.

    Parameters:
    - base_dir (str): The base directory of the CRWU dataset.
    - window_duration (int/float) : Duration of window in seconds
    - overlap (float) : Overlap ratio (0 till 1)
    - classification_type (str) : binary/multiclass (see function deduce_labels_crwu)

    Returns:
    - features_data (pd.DataFrame): Extracted features for each window.
    - time_series_data (pd.DataFrame): Corresponding time series for each window.
    """

    features_data = []
    time_series_data = []
    window_size = window_duration * fs
    step_size = int(window_size * (1 - overlap))

    # Traverse directories and process files
    for base_dir in base_dir_list:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    mat_data = loadmat(file_path, simplify_cells=True)
                    # print('file path:', file_path)

                    # Extract label and bearing ID
                    folder_name = os.path.basename(root) # 'Normal' for example
                    recording_id = file.split('.')[0]
                    # print('recording_id:', recording_id)
                    label = deduce_labels_crwu(folder_name, recording_id, classification_type)
                    # bearing_id = recording_id.split('_')[0]
                    
                    # Identify time series fields
                    time_series_keys = [key for key in mat_data.keys() if "DE_time" in key]  # or "FE_time" in key
                    # print(time_series_keys)

                    for key in time_series_keys:
                        time_series = mat_data[key]
                        
                        # Process time series into overlapping windows
                        # total_windows = max(0, (len(time_series) - window_size) // step_size + 1)
                        for start in range(0, len(time_series) - window_size + 1, step_size):
                            window = time_series[start:start + window_size]

                            # Extract features from the window
                            features = extract_features(window, prefix=f"{key.split('_')[-2]}_")
                            features['label'] = label
                            features['bearing_id'] = recording_id

                            features_data.append(features)

                            # Store time series data
                            time_series_entry = {
                                'time_series': window,
                                'label': label,
                                'bearing_id': recording_id
                            }
                            time_series_data.append(time_series_entry)

        # Convert lists to DataFrames
        features_df = pd.DataFrame(features_data)
        time_series_df = pd.DataFrame(time_series_data)
    return features_df, time_series_df
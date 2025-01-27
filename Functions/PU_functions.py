import os
import pandas as pd

from scipy.io import loadmat

from general_functions import *

def deduce_labels_pu(bearing_id, classification_type):
    """
    Deduce labels for a given bearing_id based on the classification type.
    
    Parameters:
    - bearing_id (str): The ID of the bearing (for example 'KA01', 'KA04').
    - classification_type (str): The type of classification ('binary', 'multiclass')
    
    Returns:
    - int: The label for the given bearing_id.
    """
    
    # Real damages labels (from Table 5)
    real_damage_labels = {
        "KA04": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "OR", "characteristic": "Single Point"},
        "KA15": {"fault_type": "Plastic Deformation: Indentations", "extent": "Single", "component": "OR", "characteristic": "Single Point"},
        "KA16": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "OR", "characteristic": "Single Point"},
        "KA22": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "OR", "characteristic": "Single Point"},
        "KA30": {"fault_type": "Plastic Deformation: Indentations", "extent": "Distributed", "component": "OR", "characteristic": "Distributed"},
        "KB23": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "Balls", "characteristic": "Single Point"}, #+IR
        "KB24": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "Balls", "characteristic": "Single Point"}, #+IR
        "KB27": {"fault_type": "Plastic Deformation: Indentations", "extent": "Distributed", "component": "Balls", "characteristic": "Distributed"},# OR+
        "KI04": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
        "KI14": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
        "KI16": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
        "KI17": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
        "KI18": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
        "KI21": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"}
    }

        
    # # Real damages labels (from Table 5)
    # real_damage_labels = {
    #     "KA04": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "OR", "characteristic": "Single Point"},
    #     "KA15": {"fault_type": "Plastic Deformation: Indentations", "extent": "Single", "component": "OR", "characteristic": "Single Point"},
    #     "KA16": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "OR", "characteristic": "Single Point"},
    #     "KA22": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "OR", "characteristic": "Single Point"},
    #     "KA30": {"fault_type": "Plastic Deformation: Indentations", "extent": "Distributed", "component": "OR", "characteristic": "Distributed"},
    #     "KB23": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "OR", "characteristic": "Single Point"}, #+IR
    #     "KB24": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "OR", "characteristic": "Single Point"}, #+IR
    #     "KB27": {"fault_type": "Plastic Deformation: Indentations", "extent": "Distributed", "component": "IR", "characteristic": "Distributed"},# OR+
    #     "KI04": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
    #     "KI14": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
    #     "KI16": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
    #     "KI17": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
    #     "KI18": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"},
    #     "KI21": {"fault_type": "Fatigue: Pitting", "extent": "Single", "component": "IR", "characteristic": "Single Point"}
    # }
    
    # Artificial damages labels (from Table 4)
    artificial_damage_labels = {
        "KA01": {"damage_level": 1, "method": "EDM"},
        "KA03": {"damage_level": 2, "method": "electric engraver"},
        "KA05": {"damage_level": 2, "method": "electric engraver"},
        "KA06": {"damage_level": 2, "method": "drilling"},
        "KA07": {"damage_level": 2, "method": "drilling"},
        "KA08": {"damage_level": 2, "method": "electric engraver"},
        "KA09": {"damage_level": 2, "method": "EDM"},
        "KI01": {"damage_level": 1, "method": "electric engraver"},
        "KI03": {"damage_level": 2, "method": "electric engraver"},
        "KI05": {"damage_level": 2, "method": "electric engraver"},
        "KI07": {"damage_level": 2, "method": "electric engraver"},
        "KI08": {"damage_level": 2, "method": "electric engraver"}
    }

    # Bearing_type is either 'Real Damages', 'Artificially Damaged' or Healthy
    if bearing_id in real_damage_labels:
        bearing_type = 'Real Damages'
    elif bearing_id in artificial_damage_labels:
        bearing_type = 'Artificially Damaged'
    else:
        bearing_type = 'Healthy'

    if ((bearing_id not in real_damage_labels) and (bearing_id not in artificial_damage_labels) and (bearing_type != "Healthy")):
        raise ValueError(f"Bearing ID {bearing_id} not found in real damage labels.")

    # Classification
    ## BINARY
    if classification_type == "binary":
        return 0 if bearing_type == "Healthy" else 1
    
    ## MULTICLASS
    elif classification_type == "multiclass":
        labels = real_damage_labels if bearing_type == "Real Damages" else artificial_damage_labels
        if bearing_type == "Healthy":
            return 0
        else:
            component_map = {"OR": 1, "IR": 2, "Balls":3}
            return component_map[labels[bearing_id]['component']]
    

def load_pu_dataset(base_dir_list, window_duration, overlap, signal_indices, classification_type="binary"):
    """
    Load the Paderborn University (PU) dataset, process time series into windows (with overlap), 
    and extract features. 

    Parameters:
    - base_dir_list (list): Directories containing bearing folders of interest
    - window_duration (int/float) : Duration of window in seconds
    - overlap (float) : Overlap ratio (0 till 1)
    - signal_indices (list) : Time series which will be used (for example [1,2,6])
        (0) : 'force'           
        (1) : 'phase_current_1'
        (2) : 'phase_current_2'
        (3) : 'speed'
        (4) : 'temp_2_bearing_module'
        (5) : 'torque'
        (6) : 'vibration_1'
    - classification_type (str) : binary/multiclass (see function deduce_labels_pu)

    Returns:
    - features (dict): Features and their associated labels
    - time_series (dict): Time series and their associated labels
    """
    features_data = pd.DataFrame()
    time_series_data = pd.DataFrame()
    
    # Traverse directories and process files
    for base_dir in base_dir_list:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    mat_data = loadmat(file_path, simplify_cells=True)
                    
                    # Extract label and bearing ID
                    bearing_id = file.split('_')[3]
                    label = deduce_labels_pu(bearing_id, classification_type)
                    
                    # Extract time series
                    file_name = file.split('.')[0]
                    ts_data = mat_data[file_name]['Y'] if 'Y' in mat_data[file_name] else ValueError('There is no Y value')

                    # Initialize feature and time series dictionaries
                    feature_dicts = []
                    time_series_dicts = []

                    for idx, signal in enumerate(ts_data):
                        # Iterate through all signal_indices of interest and update the dictonary in every loop
                        if idx in signal_indices:
                            time_series = signal['Data']
                            sample_rate = extract_sample_rate(signal['Raster'])  # Extract sample rate dynamically

                            window_size = int(window_duration * sample_rate)
                            step_size = int(window_size * (1 - overlap))
                            total_windows = max(0, (len(time_series) - window_size) // step_size + 1) # Total windows needed for the dicts
                            
                            if signal_indices[0] == idx:    # Needs to be done only once
                                feature_dicts.extend([{} for _ in range(total_windows)])
                                time_series_dicts.extend([{} for _ in range(total_windows)])
                            
                            feature_dicts, time_series_dicts = process_time_series(time_series, window_size, overlap, f'{idx}_', feature_dicts, time_series_dicts)

                    # Transform retrieved dictonairies of file to pandas dataframes
                    feature_dicts= pd.DataFrame(feature_dicts)
                    time_series_dicts= pd.DataFrame(time_series_dicts)
                    
                    # Add labels
                    feature_dicts["bearing_id"] = bearing_id
                    feature_dicts["label"] = label
                    time_series_dicts["bearing_id"] = bearing_id
                    time_series_dicts["label"] = label

                    # Concatenate the found dataframes
                    if features_data.empty:               
                        features_data = pd.DataFrame(feature_dicts)
                        time_series_data = pd.DataFrame(time_series_dicts)
                    else:
                        features_data = pd.concat([features_data, feature_dicts], ignore_index=True)
                        time_series_data = pd.concat([time_series_data, time_series_dicts], ignore_index=True)

                    
    return features_data, time_series_data


def extract_sample_rate(raster):
    """
    Extract the sample rate from the 'Raster' field,
    """
    if isinstance(raster, np.ndarray):
        print(raster)
        raster = raster.item()  # Extract the single element if it's a scalar array
    rate_mapping = {
        "Mech_4kHz": 4000,
        "HostService": 62000,
        "Temp_1Hz": 1
    }
    return rate_mapping[raster]
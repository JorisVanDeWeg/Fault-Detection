import numpy as np
from scipy.stats import skew, kurtosis, entropy

# def extract_features(data, prefix):
#     """
#     Extract features for a given time series.

#     Parameters:
#     - data (list): Time series of interest
#     - prefix (str): Prefix used in dictonary for clarity

#     Returns:
#     - features (dict): Features and their associated values
#     """
    
#     features = {
#         f'{prefix}mean'        : np.mean(data),
#         f'{prefix}std'         : np.std(data),
#         f'{prefix}pulse'       : np.max(data) - np.mean(data), # np.max(data) / np.mean(data) if np.mean(data) != 0 else 0,
#         f'{prefix}peak_to_peak': np.max(data) - np.min(data),
#         f'{prefix}kurtosis'    : kurtosis(data),
#         f'{prefix}skewness'    : skew(data),
#     }
#     return features


def extract_features(data, prefix):
    """
    Extract features for a given time series.

    Parameters:
    - data (list): Time series of interest
    - prefix (str): Prefix used in dictionary for clarity

    Returns:
    - features (dict): Features and their associated values
    """
    data = np.array(data)  # Ensure data is a numpy array
    
    # Root Mean Square (RMS)
    rms = np.sqrt(np.mean(np.square(data)))
    
    # Clearance Factor: max(data) / mean(abs(data))^2
    clearance_factor = np.max(data) / (np.mean(np.abs(data)) ** 2)
    
    # Crest Factor: max(data) / RMS
    crest_factor = np.max(data) / rms if rms != 0 else 0
    
    # Entropy
    hist, bin_edges = np.histogram(data, bins=10, density=True)
    prob_density = hist / np.sum(hist)
    signal_entropy = entropy(prob_density)

    features = {
        f'{prefix}mean'        : np.mean(data),
        f'{prefix}std'         : np.std(data),
        f'{prefix}pulse'       : np.max(data) - np.mean(data),
        f'{prefix}peak_to_peak': np.max(data) - np.min(data),
        f'{prefix}kurtosis'    : kurtosis(data),
        f'{prefix}skewness'    : skew(data),
        f'{prefix}rms'         : rms,
        f'{prefix}clearance_factor': clearance_factor,
        f'{prefix}crest_factor': crest_factor,
        f'{prefix}entropy'     : signal_entropy,
    }
    return features
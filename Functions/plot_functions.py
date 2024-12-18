import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_combined_results(dataframes, labels, metric='test_accuracy', title='Performance Comparison'):
    """
    Combines multiple DataFrames and plots the performance of classifiers across methods.

    Parameters:
    - dataframes: list of pd.DataFrame, results from different methods.
    - labels: list of str, labels for each method (e.g., 'Random Split', 'Group Split').
    - metric: str, the column name for the performance metric to plot ('test_accuracy' or 'f1_score').
    - title: str, the title for the plot.
    """
    # Validate inputs
    if len(dataframes) != len(labels):
        raise ValueError("Number of DataFrames and labels must be the same.")

    # Combine all DataFrames into one, adding a 'method' column
    combined_df = pd.concat(
        [df.assign(method=label) for df, label in zip(dataframes, labels)],
        ignore_index=True
    )

    # Plot the combined results
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=combined_df, 
        x='model', 
        y=metric, 
        hue='method', 
        ci=None  # Disable confidence intervals for clarity
    )
    plt.title(title)
    plt.ylabel(metric.replace('_', ' ').title())  # Format y-axis label
    plt.xlabel('Classifier')
    plt.legend(title='Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_combined_results2(dataframes, labels, file_name, metric='test_accuracy', title='Performance Comparison', plot_type='boxplot'):
    """
    Combines multiple DataFrames and plots the performance of classifiers across methods.

    Parameters:
    - dataframes: list of pd.DataFrame, results from different methods.
    - labels: list of str, labels for each method (e.g., 'Random Split', 'Group Split').
    - metric: str, the column name for the performance metric to plot ('test_accuracy' or 'f1_score').
    - title: str, the title for the plot.
    - plot_type: str, type of plot to use ('boxplot' or 'stripplot').
    """

    results_dir = 'Results/'

    # Validate inputs
    if len(dataframes) != len(labels):
        raise ValueError("Number of DataFrames and labels must be the same.")

    # Combine all DataFrames into one, adding a 'method' column
    combined_df = pd.concat(
        [df.assign(method=label) for df, label in zip(dataframes, labels)],
        ignore_index=True
    )

    # Plot the combined results
    plt.figure(figsize=(12, 6))
    if plot_type == 'boxplot':
        sns.boxplot(
            data=combined_df, 
            x='model', 
            y=metric, 
            hue='method', 
            showmeans=True
        )
    elif plot_type == 'stripplot':
        sns.stripplot(
            data=combined_df, 
            x='model', 
            y=metric, 
            hue='method', 
            dodge=True, 
            jitter=True, 
            size=8
        )
    else:
        raise ValueError("Invalid plot_type. Use 'boxplot' or 'stripplot'.")
    
    plt.title(title)
    plt.ylabel(metric.replace('_', ' ').title())  # Format y-axis label
    plt.xlabel('Classifier')
    plt.legend(title='Method')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(results_dir, f"{file_name}.png")
    plt.savefig(plot_filename)
    plt.close()

    # Compute mean +/- std for each method and model
    summary = combined_df.groupby(['model', 'method'])[metric].agg(['mean', 'std']).reset_index()
    summary['formatted'] = summary.apply(lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)

    # Save the formatted results to a text file (and print)
    results_filename = os.path.join(results_dir, f"{file_name}_summary.txt")
    # print("\nPerformance Summary (Mean ± Std):")
    with open(results_filename, 'w') as file:
        file.write("Performance Summary (Mean ± Std):\n")
        for method in labels:
            file.write(f"\nMethod: {method}\n")
            # print(f"\nMethod: {method}")
            method_summary = summary[summary['method'] == method]
            for _, row in method_summary.iterrows():
                file.write(f"{row['model']}: {row['formatted']}\n")
                # print(f"{row['model']}: {row['formatted']}")
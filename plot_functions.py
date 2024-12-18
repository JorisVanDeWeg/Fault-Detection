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


def plot_combined_results2(dataframes, labels, metric='test_accuracy', title='Performance Comparison', plot_type='boxplot'):
    """
    Combines multiple DataFrames and plots the performance of classifiers across methods.

    Parameters:
    - dataframes: list of pd.DataFrame, results from different methods.
    - labels: list of str, labels for each method (e.g., 'Random Split', 'Group Split').
    - metric: str, the column name for the performance metric to plot ('test_accuracy' or 'f1_score').
    - title: str, the title for the plot.
    - plot_type: str, type of plot to use ('boxplot' or 'stripplot').
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
    plt.show()

    # Compute mean +/- std for each method and model
    summary = combined_df.groupby(['model', 'method'])[metric].agg(['mean', 'std']).reset_index()
    summary['formatted'] = summary.apply(lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)
    
    # Print formatted results
    print("\nPerformance Summary (Mean ± Std):")
    for method in labels:
        print(f"\nMethod: {method}")
        method_summary = summary[summary['method'] == method]
        for _, row in method_summary.iterrows():
            print(f"{row['model']}: {row['formatted']}")


# def print_results(results_df):
#     # Group by 'model' and compute mean and standard deviation
#     model_summary = results_df.groupby('model').agg({
#         'test_accuracy': ['mean', 'std'],
#         'f1_score': ['mean', 'std']
#     })

#     # Combine mean and std into "mean ± std" format
#     model_summary['Accuracy'] = model_summary[('test_accuracy', 'mean')].round(4).astype(str) + " ± " + model_summary[('test_accuracy', 'std')].round(4).astype(str)
#     model_summary['F1-Score'] = model_summary[('f1_score', 'mean')].round(4).astype(str) + " ± " + model_summary[('f1_score', 'std')].round(4).astype(str)

#     # Select relevant columns and reset index
#     final_summary = model_summary[['Accuracy', 'F1-Score']].reset_index()

#     # Display the formatted summary
#     print(final_summary)

#     # Optionally save to CSV
#     final_summary.to_csv('model_performance_summary.csv', index=False)
#     return

# plt.figure(figsize=(10, 5))
# sns.boxplot(data=final_results_pu, x='model', y='test_accuracy')
# plt.title('Test Accuracy Across Folds (PU)')
# plt.ylabel('Accuracy')
# plt.xlabel('Model')
# plt.xticks(rotation=45)
# plt.show()

# plt.figure(figsize=(10, 5))
# sns.boxplot(data=final_results_pu, x='model', y='f1_score')
# plt.title('F1 Score Across Folds (PU)')
# plt.ylabel('F1 Score')
# plt.xlabel('Model')
# plt.xticks(rotation=45)
# plt.show()
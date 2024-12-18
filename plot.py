import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

final_results_pu = pd.read_csv('final_results_pu.csv')

plt.figure(figsize=(10, 5))
sns.boxplot(data=final_results_pu, x='model', y='test_accuracy')
plt.title('Test Accuracy Across Folds (PU)')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=final_results_pu, x='model', y='f1_score')
plt.title('F1 Score Across Folds (PU)')
plt.ylabel('F1 Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()
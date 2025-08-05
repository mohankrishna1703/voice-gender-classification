import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("data/vocal_gender_features_new.csv")

# Count plot: How many male (1) vs female (0)
sns.countplot(x='label', data=data)
plt.title("Gender Distribution (0 = Female, 1 = Male)")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Correlation heatmap: See which features are related
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()
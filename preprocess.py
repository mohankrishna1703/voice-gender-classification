import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset with the correct path
data = pd.read_csv("data/vocal_gender_features_new.csv")

# Separate features (X) and labels (y)
X = data.drop("label", axis=1)  # all columns except 'label'
y = data["label"]               # the target column

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data has been normalized! Here’s what it looks like (first 5 rows):")
print(X_scaled[:5])
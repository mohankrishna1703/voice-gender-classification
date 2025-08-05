import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv("data/vocal_gender_features_new.csv")

# Separate features
X = data.drop("label", axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train KMeans model
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

print("First 10 cluster labels:", clusters[:10])
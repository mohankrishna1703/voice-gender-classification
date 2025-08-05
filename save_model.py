import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = pd.read_csv("data/vocal_gender_features_new.csv")
X = data.drop("label", axis=1)
y = data["label"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save model to file
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model saved as rf_model.pkl")
import pandas as pd

# Step 1: Load the dataset
data = pd.read_csv("data/vocal_gender_features_new.csv")  # make sure your file is named like this!

# Step 2: See the first few rows
print("Here's how your data looks:")
print(data.head())

# Step 3: Check for missing values
print("\nAre there any missing values?")
print(data.isnull().sum())
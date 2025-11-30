import pandas as pd
from sklearn.cluster import KMeans
import pickle

# =====================================================
# 1. Load dataset using correct separator
# =====================================================
df = pd.read_csv("winequality-red.csv", sep=';')

# Keep only numeric columns
df_numeric = df.select_dtypes(include=['float64', 'int64'])

print("Numeric columns found:", df_numeric.columns)

# =====================================================
# 2. Train K-Means (k=3)
# =====================================================
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_numeric)

print("K-Means model trained with k=3")

# =====================================================
# 3. Save MODEL as PKL
# =====================================================
model_filename = "kmeans_wine_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(kmeans, f)

print(f"Model saved as {model_filename}")

# =====================================================
# 4. Save FEATURES as PKL
# =====================================================
features_filename = "wine_features.pkl"
with open(features_filename, "wb") as f:
    pickle.dump(list(df_numeric.columns), f)

print(f"Features saved as {features_filename}")

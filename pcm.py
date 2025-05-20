#pcm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset — FIXED PATH issue using raw string
df = pd.read_csv(r"C:\Users\HP\Downloads\heart.csv")

# Optional: Preview your dataset
print("First 5 rows of data:")
print(df.head())

# Handle non-numeric columns (e.g., gender M/F) — Convert categorical to numeric
df_encoded = pd.get_dummies(df, drop_first=True)

# Extract features
X = df_encoded.values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCM Clustering implementation
def initialize_clusters(X, n_clusters):
    np.random.seed(0)
    indices = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[indices]
    return centers

def calculate_distance(X, centers):
    return np.linalg.norm(X[:, np.newaxis] - centers, axis=2)

def update_membership(X, centers, m):
    dist = calculate_distance(X, centers)
    dist = np.fmax(dist, 1e-10)  # Avoid divide by zero
    exponent = 2 / (m - 1)
    tmp = (dist[:, :, np.newaxis] / dist[:, np.newaxis, :]) ** exponent
    U = 1 / np.sum(tmp, axis=2)
    return U

def update_centers(X, U, m):
    um = U ** m
    return (um.T @ X) / np.sum(um.T, axis=1)[:, np.newaxis]

def pcm(X, n_clusters=3, m=2, max_iter=100, tol=1e-5):
    centers = initialize_clusters(X, n_clusters)
    for i in range(max_iter):
        U = update_membership(X, centers, m)
        new_centers = update_centers(X, U, m)
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    return centers, U

# Run PCM clustering
n_clusters = 3
m = 2
centers, U = pcm(X_scaled, n_clusters=n_clusters, m=m)
labels = np.argmax(U, axis=1)

# Plotting clusters
plt.figure(figsize=(8, 6))
for cluster in np.unique(labels):
    plt.scatter(X_scaled[labels == cluster, 0], X_scaled[labels == cluster, 1], label=f'Cluster {cluster+1}')

plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centers')
plt.title('PCM Clustering Results')
plt.xlabel('Feature 1 (standardized)')
plt.ylabel('Feature 2 (standardized)')
plt.legend()
plt.show()

# Silhouette Score calculation with safe check
unique_clusters = np.unique(labels)
n_clusters_found = len(unique_clusters)
print(f"Unique clusters found: {n_clusters_found}")

if n_clusters_found > 1:
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {score:.3f}")
else:
    print(f"Only {n_clusters_found} cluster found — Silhouette Score cannot be computed.")


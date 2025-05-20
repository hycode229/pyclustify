#pfcm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset (FIXED path)
df = pd.read_csv(r"C:\Users\HP\Downloads\heart.csv")

# Preview dataset
print("First 5 rows of data:")
print(df.head())

# Handle non-numeric columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Extract features
X = df_encoded.values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize cluster centers
def initialize_clusters(X, n_clusters):
    np.random.seed(0)
    indices = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[indices]
    return centers

# Euclidean distance calculation
def calculate_distance(X, centers):
    return np.linalg.norm(X[:, np.newaxis] - centers, axis=2)

# PFCM function
def pfcm(X, n_clusters=3, m=2, eta=2, a=1, b=1, max_iter=100, tol=1e-5):
    N, d = X.shape
    centers = initialize_clusters(X, n_clusters)
    U = np.random.dirichlet(np.ones(n_clusters), size=N).T  # shape: (n_clusters, N)
    T = np.copy(U)

    for it in range(max_iter):
        dist = calculate_distance(X, centers)  # shape: (N, n_clusters)
        dist = np.fmax(dist, 1e-10)

        # Update U (Fuzzy Membership Matrix)
        exponent = 2 / (m - 1)
        tmp = (dist[:, :, np.newaxis] / dist[:, np.newaxis, :]) ** exponent
        U_new = 1 / np.sum(tmp, axis=2).T

        # Update T (Typicality Matrix)
        T_new = np.exp(-(dist.T ** 2) / eta)

        # Combine U and T using weighting factors a and b
        um = U_new ** m
        tm = T_new ** eta
        numerator = a * um + b * tm
        centers_new = (numerator @ X) / np.sum(numerator, axis=1)[:, np.newaxis]

        # Convergence check
        if np.linalg.norm(centers_new - centers) < tol:
            break

        centers = centers_new
        U = U_new
        T = T_new

    labels = np.argmax(U, axis=0)
    return centers, U, T, labels

# Run PFCM clustering
n_clusters = 3
m = 2
eta = 2
a = 1
b = 1
centers, U, T, labels = pfcm(X_scaled, n_clusters=n_clusters, m=m, eta=eta, a=a, b=b)

# Plotting clusters
plt.figure(figsize=(8, 6))
for cluster in np.unique(labels):
    plt.scatter(X_scaled[labels == cluster, 0], X_scaled[labels == cluster, 1], label=f'Cluster {cluster+1}')

plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centers')
plt.title('PFCM Clustering Results')
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


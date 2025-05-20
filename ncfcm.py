#ncfcm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\heart.csv")
print("Dataset shape:", df.shape)

# Encode categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Prepare features
X = df_encoded.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Parameters
n_clusters = 3  # main clusters (excluding noise)
m = 2.0  # fuzziness
max_iter = 150
epsilon = 1e-5
delta = 0.1  # noise distance parameter

# Initialize membership matrix randomly (including noise cluster)
np.random.seed(42)
n_samples = X_scaled.shape[0]
u = np.random.dirichlet(np.ones(n_clusters + 1), size=n_samples)

# Main NFCM loop
for iteration in range(max_iter):
    um = u[:, :n_clusters] ** m  # exclude noise for center updates
    centers = um.T @ X_scaled / np.sum(um, axis=0)[:, None]

    # Compute distances
    dist = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        dist[:, i] = np.linalg.norm(X_scaled - centers[i], axis=1)
    dist = np.fmax(dist, 1e-10)

    # Add noise distance term
    d_noise = np.full(n_samples, delta)

    # Update membership matrix (including noise)
    u_new = np.zeros((n_samples, n_clusters + 1))
    for k in range(n_clusters):
        u_new[:, k] = (dist[:, k]) ** (-2 / (m - 1))
    u_new[:, -1] = (d_noise) ** (-2 / (m - 1))  # noise cluster

    u_new_sum = np.sum(u_new, axis=1, keepdims=True)
    u_new /= u_new_sum

    # Check convergence
    if np.linalg.norm(u_new - u) < epsilon:
        print(f"Converged at iteration {iteration}")
        break
    u = u_new

# Assign hard labels (including noise as cluster -1)
labels = np.argmax(u, axis=1)
labels[labels == n_clusters] = -1  # mark noise cluster as -1

# Visualize clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i}')
plt.scatter(X_pca[labels == -1, 0], X_pca[labels == -1, 1], c='black', label='Noise', marker='x')
plt.title('Noise Clustering Fuzzy C-Means (NFCM)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Show memberships
membership_df = pd.DataFrame(u, columns=[f'Cluster_{i}' for i in range(n_clusters)] + ['Noise'])
print("\nSample memberships (first 5 rows):")
print(membership_df.head())

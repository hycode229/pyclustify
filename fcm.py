#fcm
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

# Prepare features (can exclude 'HeartDisease' if desired)
X = df_encoded.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Parameters
n_clusters = 3
m = 2.0  # fuzziness parameter
max_iter = 150
epsilon = 1e-5

# Initialize membership matrix randomly
np.random.seed(42)
n_samples = X_scaled.shape[0]
u = np.random.dirichlet(np.ones(n_clusters), size=n_samples)

# Main FCM loop
for iteration in range(max_iter):
    # Compute cluster centers
    um = u ** m
    centers = um.T @ X_scaled / np.sum(um, axis=0)[:, None]

    # Compute distances
    dist = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        dist[:, i] = np.linalg.norm(X_scaled - centers[i], axis=1)
    dist = np.fmax(dist, 1e-10)  # avoid division by zero

    # Update membership matrix
    u_new = 1.0 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (2 / (m - 1)), axis=2)

    # Check convergence
    if np.linalg.norm(u_new - u) < epsilon:
        print(f"Converged at iteration {iteration}")
        break
    u = u_new

# Assign hard labels
labels = np.argmax(u, axis=1)

# Visualize clusters (optional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i}')
plt.title('Fuzzy C-Means Clustering (NumPy Implementation)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Show membership matrix
membership_df = pd.DataFrame(u, columns=[f'Cluster_{i}' for i in range(n_clusters)])
print("\nSample memberships (first 5 rows):")
print(membership_df.head())

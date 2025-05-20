#cfcm
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

# Prepare features (exclude HeartDisease if desired)
X = df_encoded.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Parameters
n_clusters = 3
m = 2.0  # fuzziness
max_iter = 150
epsilon = 1e-5

# Initialize membership matrix randomly
np.random.seed(42)
n_samples = X_scaled.shape[0]
u = np.random.dirichlet(np.ones(n_clusters), size=n_samples)

# FCM loop
for iteration in range(max_iter):
    um = u ** m
    centers = um.T @ X_scaled / np.sum(um, axis=0)[:, None]

    dist = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        dist[:, i] = np.linalg.norm(X_scaled - centers[i], axis=1)
    dist = np.fmax(dist, 1e-10)

    u_new = 1.0 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (2 / (m - 1)), axis=2)

    if np.linalg.norm(u_new - u) < epsilon:
        print(f"Converged at iteration {iteration}")
        break
    u = u_new

# Simulate interval fuzzy membership (± α% uncertainty)
alpha = 0.1  # 10% uncertainty
u_lower = np.clip(u - alpha * u, 0, 1)
u_upper = np.clip(u + alpha * u, 0, 1)

# Hard labels
labels = np.argmax(u, axis=1)

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i}')
plt.title('Interval Fuzzy C-Means (IFCM) Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Display interval membership matrix
membership_interval_df = pd.DataFrame()
for i in range(n_clusters):
    membership_interval_df[f'Cluster_{i}_Lower'] = u_lower[:, i]
    membership_interval_df[f'Cluster_{i}_Upper'] = u_upper[:, i]

print("\nSample interval memberships (first 5 rows):")
print(membership_interval_df.head())

import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

PROJECT_PATH = "/content/project"
DATASET_FOLDER = os.path.join(PROJECT_PATH, "dataset_v2")

X_path = os.path.join(DATASET_FOLDER, "X.npy")
Y_path = os.path.join(DATASET_FOLDER, "Y_ml.npy")
G_path = os.path.join(DATASET_FOLDER, "groups.npy")

print("Loading dataset_v2...")
X = np.load(X_path)
Y = np.load(Y_path)
groups = np.load(G_path, allow_pickle=True)

assert X.shape[0] == Y.shape[0] == groups.shape[0], "Sample count mismatch!"

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"groups shape: {groups.shape}")

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

print("Fitting scalers...")
X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

np.save(os.path.join(DATASET_FOLDER, "X_scaled.npy"), X_scaled)
np.save(os.path.join(DATASET_FOLDER, "Y_scaled.npy"), Y_scaled)

with open(os.path.join(DATASET_FOLDER, "x_scaler.pkl"), "wb") as f:
    pickle.dump(x_scaler, f)
with open(os.path.join(DATASET_FOLDER, "y_scaler.pkl"), "wb") as f:
    pickle.dump(y_scaler, f)

print("Saved:")
print(" - X_scaled.npy, Y_scaled.npy")
print(" - x_scaler.pkl, y_scaler.pkl")
print("\n[02_scale_dataset_v2.py DONE]")

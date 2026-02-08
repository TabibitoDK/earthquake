import os
import numpy as np
import pickle
import tensorflow as tf

PROJECT_PATH = "/content/project"
DATASET_FOLDER = os.path.join(PROJECT_PATH, "dataset_v2")
MODEL_FOLDER = os.path.join(PROJECT_PATH, "models_v2")

with open(os.path.join(DATASET_FOLDER, "x_scaler.pkl"), "rb") as f:
    x_scaler = pickle.load(f)
with open(os.path.join(DATASET_FOLDER, "y_scaler.pkl"), "rb") as f:
    y_scaler = pickle.load(f)

# ---- theory constants
Dp = 354.0
H = 523.0
g = 980.0  # not needed here, but keep consistent

def calc_r(Dp):
    if Dp <= 354.0:
        return 0.001214 * Dp + 0.5698
    else:
        return 1.0

def calc_rho(E_Nmm, Vp, Dp):
    r = calc_r(Dp)
    E_over_Vp = E_Nmm / Vp
    E_eff_over_Vp = r * E_over_Vp
    rho = (8.33 / 7.967) * (-0.06 + 1.25 * np.exp(-E_eff_over_Vp / 360.0))
    return min(rho, 1.0)

def extract_features(t, ag):
    dt = t[1] - t[0]
    PGA = np.max(np.abs(ag))
    PGV = np.max(np.abs(np.cumsum(ag) * dt))
    PGD = np.max(np.abs(np.cumsum(np.cumsum(ag) * dt) * dt))
    RMS = np.sqrt(np.mean(ag ** 2))
    IA = np.trapezoid(ag ** 2, dx=dt)
    CAV = np.sum(np.abs(ag)) * dt
    duration = t[-1] - t[0]
    return np.array([PGA, PGV, PGD, RMS, IA, CAV, duration], dtype=float)

print("\n=== PREDICTION MODE (v2) ===")

W_total = float(input("Enter building weight W_total [kN]: ") or "8437.5")
floors = int(input("Enter number of floors (5, 10, 20, 30): ") or "10")
h = float(input("Enter damping ratio h (e.g. 0.02): ") or "0.02")
alpha = float(input("Enter alpha value (0.05, 0.06, 0.065): ") or "0.05")

if floors not in [5, 10, 20, 30]:
    raise ValueError("floors must be 5, 10, 20, or 30")

eq_path = input("Enter earthquake file path: ").strip()
if not os.path.exists(eq_path):
    raise FileNotFoundError("Earthquake file not found")

scale = float(input("Enter scaling factor (e.g. 1.0, 1.67): ") or "1.0")

# load EQ
t, ag0 = np.loadtxt(eq_path, unpack=True)
ag = ag0 * scale

# build X (note scale is an input feature now)
features = extract_features(t, ag)
X = np.concatenate([[W_total, floors, h, alpha, scale], features]).reshape(1, -1)
X_scaled = x_scaler.transform(X)

# load the correct model
model_path = os.path.join(MODEL_FOLDER, f"model_f{floors}.keras")
model = tf.keras.models.load_model(model_path, compile=False)
print(f"Loaded model: {model_path}")

# predict
Y_scaled = model.predict(X_scaled, verbose=0)
Y = y_scaler.inverse_transform(Y_scaled)[0]

# decode ML outputs (63)
max_disp = Y[0:30]
max_acc  = Y[30:60]
iso_disp = Y[60]
iso_acc  = Y[61]
E        = Y[62]   # kN·cm (same as your teacher output)

# hybrid physics outputs
SigmaW = W_total * 1000.0
Vp = alpha * SigmaW * H / 8.33
E_Nmm = E * 1e4
Evp = E_Nmm / Vp
rho = calc_rho(E_Nmm, Vp, Dp)

print("\n=== ML PREDICTION RESULT (v2) ===")

print("\n--- Floor Maximum Displacement [cm] ---")
for i in range(floors):
    print(f"Floor {i+1:2d}: {max_disp[i]:.5f}")

print("\n--- Floor Maximum Absolute Acceleration [cm/s²] ---")
for i in range(floors):
    print(f"Floor {i+1:2d}: {max_acc[i]:.5f}")

print("\n--- Isolator Response ---")
print(f"Isolator max displacement = {iso_disp:.5f} cm")
print(f"Isolator max abs accel    = {iso_acc:.5f} cm/s²")

print("\n--- Energy & Degradation Indices (Hybrid Physics) ---")
print(f"E        = {E:.6e} kN·cm")
print(f"Vp       = {Vp:.6e}  (computed)")
print(f"E / Vp   = {Evp:.6e} (computed)")
print(f"rho      = {rho:.6f} (computed)")

print("\nPrediction complete.")

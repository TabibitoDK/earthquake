import os
import numpy as np

# =========================================================
# CONFIG
# =========================================================
PROJECT_PATH = "/content/earthquake"
EQ_FOLDER = os.path.join(PROJECT_PATH, "earthquake_data")
OUT_FOLDER = os.path.join(PROJECT_PATH, "dataset_v2")
os.makedirs(OUT_FOLDER, exist_ok=True)

FLOOR_LIST = [5, 10, 20, 30]
ALPHA_LIST = [0.05, 0.06, 0.065]
W_LIST = [5000, 8000, 12000, 16000]   # total building weight [kN]
SCALE_LIST = [0.5, 0.75, 1.0, 1.25, 1.5, 1.67, 1.75]  # include your 1.67

h = 0.02
g = 980.0
T0 = 4.0

# ---- lead plug parameter (theory)
Dp = 354.0    # mm

# =========================================================
# LEAD PLUG DEGRADATION (THEORY)
# =========================================================
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

# =========================================================
# EARTHQUAKE FEATURE EXTRACTION
# =========================================================
def extract_features(t, ag):
    dt = t[1] - t[0]

    PGA = np.max(np.abs(ag))
    PGV = np.max(np.abs(np.cumsum(ag) * dt))
    PGD = np.max(np.abs(np.cumsum(np.cumsum(ag) * dt) * dt))
    RMS = np.sqrt(np.mean(ag ** 2))
    IA = np.trapezoid(ag ** 2, dx=dt)  # fixed (np.trapz deprecated)
    CAV = np.sum(np.abs(ag)) * dt
    duration = t[-1] - t[0]

    return np.array([PGA, PGV, PGD, RMS, IA, CAV, duration], dtype=float)

# =========================================================
# FULL STRUCTURAL ANALYSIS (TEACHER MODEL)
# returns: max_disp(30), max_acc(30), iso_disp, iso_acc, E
# =========================================================
def run_analysis(W_total, N_story, alpha_lp, t, ag, h):

    N = N_story + 1
    m = W_total / g
    omega0 = 2 * np.pi / T0

    steps = len(t)
    dt = t[1] - t[0]

    # ---------------------------------
    # UPPER STRUCTURE STIFFNESS
    # ---------------------------------
    def calc_upper_structure():
        T1 = N_story / 10
        Wi = np.full(N_story, W_total / N_story)
        mi = W_total / g

        alpha = np.array([np.sum(Wi[i:]) / np.sum(Wi) for i in range(N_story)])
        Ai = 1 + ((1 / np.sqrt(alpha) - alpha) * (2 * T1) / (1 + 3 * T1))
        Aalpha = Ai * alpha

        k_prime = np.zeros((N_story, N_story))
        for i in range(N_story):
            k_prime[i, i] = Aalpha[i]
            if i < N_story - 1:
                k_prime[i, i + 1] = -Aalpha[i + 1]

        Jinv = np.eye(N_story) - np.eye(N_story, k=-1)
        M = Jinv @ k_prime

        eig = np.sort(np.real(np.linalg.eigvals(M)))
        omega = np.sqrt(eig[eig > 0])

        k1 = mi * (2 * np.pi / T1) ** 2 / omega[0] ** 2
        return k1 * Aalpha

    k_story = calc_upper_structure()

    # ---------------------------------
    # FIND ISOLATOR STIFFNESS kI
    # ---------------------------------
    omega_target = 2 * np.pi / T0

    def omega1_sq(KI):
        k = np.concatenate(([KI], k_story))
        K = np.zeros((N, N))
        for i in range(N):
            K[i, i] = k[i]
            if i < N - 1:
                K[i, i + 1] = -k[i + 1]

        Jinv = np.eye(N) - np.eye(N, k=-1)
        M = (Jinv @ K) / m
        eig = np.sort(np.real(np.linalg.eigvals(M)))
        return eig[0]

    lo, hi = 0.0, 1e6
    mid = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if omega1_sq(mid) > omega_target ** 2:
            hi = mid
        else:
            lo = mid

    kI = mid
    kd = 10 * kI

    # ---------------------------------
    # MATRICES
    # ---------------------------------
    Jinv = np.eye(N)
    for i in range(1, N):
        Jinv[i, i - 1] = -1.0
    J = np.linalg.inv(Jinv)

    K = np.zeros((N, N))
    K[0, 0] = kI
    K[0, 1] = -k_story[0]
    for i in range(1, N - 1):
        K[i, i] = k_story[i - 1]
        K[i, i + 1] = -k_story[i]
    K[-1, -1] = k_story[-1]

    C_story = (2 * h / omega0) * k_story
    C = np.zeros((N, N))
    C[0, 1] = -C_story[0]
    for i in range(1, N - 1):
        C[i, i] = C_story[i - 1]
        C[i, i + 1] = -C_story[i]
    C[-1, -1] = C_story[-1]

    # ---------------------------------
    # SLIP MODEL
    # ---------------------------------
    delta = alpha_lp * N * W_total / kd

    def slip(y, v):
        return 0.25 * v * (
            2 + np.sign(y + delta) - np.sign(y - delta)
            - np.sign(v) * (np.sign(y + delta) + np.sign(y - delta))
        )

    one = np.zeros(N)
    one[0] = 1.0

    def accel(u, v, y, agi):
        ky = np.zeros(N)
        ky[0] = kd * y[0]
        rhs = C @ v + K @ u + ky
        return -Jinv @ (rhs / m) - one * agi

    # ---------------------------------
    # TIME INTEGRATION (RK2)
    # ---------------------------------
    u = np.zeros((N, steps))
    v = np.zeros((N, steps))
    y = np.zeros((N, steps))

    for i in range(steps - 1):
        a1 = accel(u[:, i], v[:, i], y[:, i], ag[i])

        u2 = u[:, i] + 0.5 * dt * v[:, i]
        v2 = v[:, i] + 0.5 * dt * a1
        y2 = y[:, i] + 0.5 * dt * np.array([slip(y[0, i], v[0, i])] + [0] * (N - 1))

        a2 = accel(u2, v2, y2, ag[i])

        u[:, i + 1] = u[:, i] + dt * (v[:, i] + 2 * v2) / 3
        v[:, i + 1] = v[:, i] + dt * (a1 + 2 * a2) / 3
        y[:, i + 1] = y[:, i] + dt * (
            np.array([slip(y[0, i], v[0, i])] + [0] * (N - 1))
            + 2 * np.array([slip(y2[0], v2[0])] + [0] * (N - 1))
        ) / 3

    # ---------------------------------
    # ABSOLUTE ACCELERATION
    # ---------------------------------
    floor_acc = np.zeros((N, steps))
    for i in range(steps):
        udd = accel(u[:, i], v[:, i], y[:, i], ag[i])
        floor_acc[:, i] = J @ udd + ag[i]

    # ---------------------------------
    # MAX RESPONSES
    # ---------------------------------
    max_disp = np.zeros(30)
    max_acc = np.zeros(30)

    disp_vals = [np.max(np.abs(u[f])) for f in range(1, N)]
    acc_vals = [np.max(np.abs(floor_acc[f])) for f in range(1, N)]

    max_disp[:len(disp_vals)] = disp_vals
    max_acc[:len(acc_vals)] = acc_vals

    iso_disp_max = np.max(np.abs(u[0]))
    iso_acc_max = np.max(np.abs(floor_acc[0]))

    # ---------------------------------
    # ENERGY E
    # ---------------------------------
    E = 0.0
    for i in range(steps - 1):
        du = u[0, i + 1] - u[0, i]
        Q1 = kI * u[0, i] + kd * y[0, i]
        Q2 = kI * u[0, i + 1] + kd * y[0, i + 1]
        E += 0.5 * (Q1 + Q2) * du

    # ---------------------------------
    # OUTPUT VECTOR for ML (63)
    # ---------------------------------
    Y_ml = np.concatenate([
        max_disp,                 # 30
        max_acc,                  # 30
        [iso_disp_max],           # 1
        [iso_acc_max],            # 1
        [E],                      # 1
    ])

    return Y_ml

# =========================================================
# DATASET GENERATION
# X includes scale as input feature
# save: X.npy, Y_ml.npy, groups.npy
# =========================================================
X_list, Y_list, groups_list = [], [], []

for file in os.listdir(EQ_FOLDER):
    if not file.endswith(".txt"):
        continue

    eq_path = os.path.join(EQ_FOLDER, file)
    t, ag0 = np.loadtxt(eq_path, unpack=True)

    for scale in SCALE_LIST:
        ag = ag0 * scale
        features = extract_features(t, ag)

        for W in W_LIST:
            for floors in FLOOR_LIST:
                for alpha in ALPHA_LIST:
                    X = np.concatenate([[W, floors, h, alpha, scale], features])
                    Y = run_analysis(W, floors, alpha, t, ag, h)

                    X_list.append(X)
                    Y_list.append(Y)
                    groups_list.append(file)  # group id = earthquake file

        print(f"Done EQ: {file} (scale={scale})")

X = np.array(X_list, dtype=float)
Y = np.array(Y_list, dtype=float)
groups = np.array(groups_list, dtype=object)

np.save(os.path.join(OUT_FOLDER, "X.npy"), X)
np.save(os.path.join(OUT_FOLDER, "Y_ml.npy"), Y)
np.save(os.path.join(OUT_FOLDER, "groups.npy"), groups)

print("\nDataset v2 generation complete.")
print(f"X shape = {X.shape} (features={X.shape[1]})")
print(f"Y_ml shape = {Y.shape} (targets={Y.shape[1]})")
print(f"groups shape = {groups.shape}")

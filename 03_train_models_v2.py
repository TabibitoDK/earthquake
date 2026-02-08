import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

np.random.seed(42)
tf.random.set_seed(42)

PROJECT_PATH = "/content/earthquake"
DATASET_FOLDER = os.path.join(PROJECT_PATH, "dataset_v2")
MODEL_FOLDER = os.path.join(PROJECT_PATH, "models_v2")
os.makedirs(MODEL_FOLDER, exist_ok=True)

X = np.load(os.path.join(DATASET_FOLDER, "X_scaled.npy"))
Y = np.load(os.path.join(DATASET_FOLDER, "Y_scaled.npy"))
groups = np.load(os.path.join(DATASET_FOLDER, "groups.npy"), allow_pickle=True)

input_dim = X.shape[1]
output_dim = Y.shape[1]

print(f"X shape={X.shape}, Y shape={Y.shape}, output_dim={output_dim}")

def build_model(input_dim, output_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    out = layers.Dense(output_dim, activation="linear")(x)
    model = models.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    return model

def train_one_floor(floor_value):
    # floors is the 2nd feature in X before scaling was applied:
    # X raw = [W, floors, h, alpha, scale] + 7 features
    # BUT we are using scaled X here, so we must filter using raw floors.
    # easiest: load raw X too and filter by raw floors.
    X_raw = np.load(os.path.join(DATASET_FOLDER, "X.npy"))
    floors_raw = X_raw[:, 1].astype(int)

    idx = np.where(floors_raw == floor_value)[0]

    Xf = X[idx]
    Yf = Y[idx]
    gf = groups[idx]

    print(f"\n=== Training for floors={floor_value} ===")
    print(f"samples = {len(idx)}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(Xf, Yf, groups=gf))

    X_train, Y_train = Xf[train_idx], Yf[train_idx]
    X_val, Y_val = Xf[val_idx], Yf[val_idx]

    model = build_model(input_dim, output_dim)

    ckpt_path = os.path.join(MODEL_FOLDER, f"model_f{floor_value}.keras")
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=400,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # save final too
    final_path = os.path.join(MODEL_FOLDER, f"model_f{floor_value}_final.keras")
    model.save(final_path)
    print(f"Saved best:  {ckpt_path}")
    print(f"Saved final: {final_path}")

for f in [5, 10, 20, 30]:
    train_one_floor(f)

print("\nAll models trained.")

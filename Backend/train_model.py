import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import tensorflow as tf

# --------------------
# CONFIG
# --------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH  = os.path.join(BASE_DIR, "dataset")
MODEL_PATH    = os.path.join(BASE_DIR, "signetra_model.h5")
BEST_PATH     = os.path.join(BASE_DIR, "signetra_best.h5")  # ✅ saves best checkpoint

# ✅ Add new labels here as you collect them
LABELS = ["HELLO", "NO", "PEACE", "STOP"] # ✅ keep sorted A-Z, match app.py

SEQUENCE_LENGTH = 30
EPOCHS          = 80       # ✅ Upgraded: more epochs + early stopping handles overfit
BATCH_SIZE      = 16       # ✅ Upgraded: larger batch = more stable gradients

print(f"\n🔧 TensorFlow version: {tf.__version__}")
print(f"📂 Dataset path: {DATASET_PATH}")
print(f"🏷️  Labels: {LABELS}\n")

# --------------------
# AUGMENTATION
# --------------------
def augment_sequence(seq):
    """
    ✅ Artificially expand dataset:
    - Random Gaussian noise
    - Random scale
    - Random time shift (roll frames)
    """
    aug = seq.copy()

    # Noise
    aug = aug + np.random.normal(0, 0.008, aug.shape)

    # Scale
    aug = aug * np.random.uniform(0.92, 1.08)

    # Time shift (±2 frames)
    shift = np.random.randint(-2, 3)
    aug   = np.roll(aug, shift, axis=0)

    return aug


# --------------------
# LOAD DATASET
# --------------------
X, y = [], []

print("📊 Loading dataset...\n")

for label in LABELS:
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder):
        print(f"  ❌ Missing: {folder}")
        continue

    count = 0
    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if not file.endswith(".npy"):
                continue

            path = os.path.join(root, file)
            try:
                data = np.load(path)

                if data.shape != (SEQUENCE_LENGTH, 63):
                    print(f"  ⚠️  Skipping {file} — shape {data.shape}")
                    continue

                # Original
                X.append(data)
                y.append(label)
                count += 1

                # ✅ Augment each sample 2x
                X.append(augment_sequence(data))
                y.append(label)

                X.append(augment_sequence(data))
                y.append(label)

            except Exception as e:
                print(f"  ❌ Error: {path} — {e}")

    print(f"  ✅ {label:10s} → {count} raw samples  ({count * 3} with augmentation)")

# --------------------
# CONVERT
# --------------------
X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n📈 Dataset shape : X={X.shape}  y={y.shape}")
print(f"🏷️  Classes found : {sorted(set(y))}\n")

# --------------------
# SAFETY CHECK
# --------------------
if len(X) == 0:
    print("❌ No data found. Collect data first.")
    exit()

missing = [l for l in LABELS if l not in set(y)]
if missing:
    print(f"❌ Missing classes in dataset: {missing}")
    exit()

# --------------------
# ENCODE LABELS
# --------------------
le          = LabelEncoder()
y_encoded   = le.fit_transform(y)
y_cat       = to_categorical(y_encoded)

print(f"🔢 Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

# --------------------
# TRAIN / TEST SPLIT
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.15,
    random_state=42,
    stratify=y_encoded      # ✅ Ensures balanced split per class
)

print(f"🔀 Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# --------------------
# MODEL — upgraded architecture
# --------------------
model = Sequential([
    # Block 1
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 63)),
    BatchNormalization(),
    Dropout(0.3),

    # Block 2
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    # Dense head
    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(len(LABELS), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --------------------
# CALLBACKS
# --------------------
callbacks = [
    # ✅ Stop early if no improvement for 15 epochs
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),

    # ✅ Reduce LR when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),

    # ✅ Always save the best model checkpoint
    ModelCheckpoint(
        BEST_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# --------------------
# TRAIN
# --------------------
print("\n🚀 Training started...\n")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# --------------------
# EVALUATE
# --------------------
print("\n📊 Evaluating on test set...\n")

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss     : {loss:.4f}")
print(f"  Test Accuracy : {acc * 100:.2f}%\n")

# Classification report
y_pred     = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true     = np.argmax(y_test, axis=1)
y_pred_lbl = le.inverse_transform(y_pred)
y_true_lbl = le.inverse_transform(y_true)

print("📋 Classification Report:")
print(classification_report(y_true_lbl, y_pred_lbl, target_names=LABELS))

# --------------------
# CONFUSION MATRIX PLOT
# --------------------
try:
    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\n📊 Confusion matrix saved: {cm_path}")
except Exception as e:
    print(f"⚠️  Could not plot confusion matrix: {e}")

# --------------------
# TRAINING CURVE PLOT
# --------------------
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'],     label='Train Acc',  color='#38bdf8')
    ax1.plot(history.history['val_accuracy'], label='Val Acc',    color='#22c55e')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train Loss', color='#38bdf8')
    ax2.plot(history.history['val_loss'], label='Val Loss',   color='#f59e0b')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(BASE_DIR, "training_curves.png")
    plt.savefig(curve_path)
    print(f"📈 Training curves saved: {curve_path}")
except Exception as e:
    print(f"⚠️  Could not plot training curves: {e}")

# --------------------
# SAVE FINAL MODEL
# --------------------
model.save(MODEL_PATH)
print(f"\n✅ Final model saved : {MODEL_PATH}")
print(f"🏆 Best model saved  : {BEST_PATH}")
print(f"\n🎉 Done! Use '{os.path.basename(BEST_PATH)}' in app.py for best accuracy.\n")

# --------------------
# IMPORTANT REMINDER
# --------------------
print("=" * 55)
print("⚠️  IMPORTANT: Update app.py LABELS to match exactly:")
print(f"   LABELS = {LABELS}")
print("   And use signetra_best.h5 instead of signetra_model.h5")
print("=" * 55)

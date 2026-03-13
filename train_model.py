import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------------------
# CONFIG
# --------------------
DATASET_PATH = "dataset"
LABELS = ["HELLO", "YES", "NO"]
SEQUENCE_LENGTH = 30

# --------------------
# Load dataset
# --------------------
X = []
y = []

for label in LABELS:
    folder_path = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder_path):
        data = np.load(os.path.join(folder_path, file))
        X.append(data)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------
# Encode labels
# --------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# --------------------
# Train-test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# --------------------
# LSTM Model
# --------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 63)))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(LABELS), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------
# Train model
# --------------------
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# --------------------
# Save model
# --------------------
model.save("signetra_model.h5")
print("Model saved as signetra_model.h5")

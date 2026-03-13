import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# --------------------
# CONFIG
# --------------------s
MODEL_PATH = "signetra_model.h5"
LABELS = ["HELLO", "YES", "NO"]
SEQUENCE_LENGTH = 30

# --------------------
# Load model
# --------------------
model = load_model(MODEL_PATH)

# --------------------
# MediaPipe setup
# --------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

sequence = []
prediction_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            frame_data = []
            for lm in hand_landmarks.landmark:
                frame_data.extend([lm.x, lm.y, lm.z])

            if len(frame_data) == 63:
                sequence.append(frame_data)

                if len(sequence) > SEQUENCE_LENGTH:
                    sequence.pop(0)

                if len(sequence) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(sequence, axis=0)
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction)
                    prediction_text = LABELS[predicted_index]

    # Display prediction
    cv2.putText(
        frame,
        f"Prediction: {prediction_text}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )

    cv2.imshow("SIGNETRA - Real Time Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

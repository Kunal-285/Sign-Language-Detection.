import cv2
import numpy as np
import os
import mediapipe as mp

# --------------------
# CONFIG — change LABEL each time you collect
# --------------------
LABEL           = "NO"     # ✅ Change this every time
SEQUENCE_LENGTH = 30
TOTAL_SEQUENCES = 200         # ✅ Upgraded: 200 sequences for better accuracy

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", LABEL)
os.makedirs(DATA_PATH, exist_ok=True)

# --------------------
# MEDIAPIPE — optimized settings
# --------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,              # ✅ Upgraded: complexity 1 for better landmark accuracy
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# --------------------
# CAMERA
# --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --------------------
# STATE
# --------------------
sequence        = []
sequence_count  = 0
collecting      = False          # ✅ New: press SPACE to start each sequence
countdown       = 0
COUNTDOWN_START = 20             # frames of countdown before recording

print(f"\n🚀 Collecting data for: [{LABEL}]")
print(f"📦 Target: {TOTAL_SEQUENCES} sequences x {SEQUENCE_LENGTH} frames")
print(f"\n⌨️  Controls:")
print(f"   SPACE → Start recording next sequence")
print(f"   ESC   → Quit early\n")

# --------------------
# HELPERS
# --------------------
def normalize_keypoints(hand_landmarks):
    """
    ✅ Normalize keypoints relative to wrist (landmark 0)
    This makes predictions position-independent
    """
    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = raw[0]
    normalized = raw - wrist  # all points relative to wrist
    return normalized.flatten()  # shape: (63,)


def draw_ui(frame, seq_count, seq_len, collecting, countdown, label):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (10, 15, 30), -1)

    # Label
    cv2.putText(frame, f"LABEL: {label}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2)

    # Sequence count
    cv2.putText(frame, f"Saved: {seq_count} / {TOTAL_SEQUENCES}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2)

    # Progress bar
    bar_w = int((seq_count / TOTAL_SEQUENCES) * (w - 20))
    cv2.rectangle(frame, (10, 55), (w - 10, 60), (30, 30, 30), -1)
    cv2.rectangle(frame, (10, 55), (10 + bar_w, 60), (34, 197, 94), -1)

    # Frame buffer bar
    if collecting and seq_len > 0:
        buf_w = int((seq_len / SEQUENCE_LENGTH) * (w - 20))
        cv2.rectangle(frame, (10, 62), (w - 10, 67), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 62), (10 + buf_w, 67), (56, 189, 248), -1)

    # Status
    if collecting:
        if countdown > 0:
            # Countdown
            num = (countdown // 7) + 1
            cv2.putText(frame, f"GET READY... {num}", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (245, 158, 11), 3)
        else:
            # Recording
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 70, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame: {seq_len}/{SEQUENCE_LENGTH}",
                        (w // 2 - 100, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
    else:
        cv2.putText(frame, "Press SPACE to record", (w // 2 - 150, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    return frame


# --------------------
# MAIN LOOP
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        hand_lms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(56, 189, 248), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(34, 197, 94), thickness=2))

        if collecting:
            if countdown > 0:
                countdown -= 1
            else:
                # ✅ Use normalized keypoints
                kp = normalize_keypoints(hand_lms)
                sequence.append(kp)

                if len(sequence) == SEQUENCE_LENGTH:
                    file_path = os.path.join(DATA_PATH, f"{sequence_count}.npy")
                    np.save(file_path, np.array(sequence))
                    print(f"  ✅ Saved sequence [{sequence_count + 1}/{TOTAL_SEQUENCES}]")

                    sequence       = []
                    sequence_count += 1
                    collecting     = False

    else:
        # Hand lost mid-sequence — reset
        if collecting and countdown == 0 and len(sequence) > 0:
            sequence   = []
            collecting = False
            print("  ⚠️  Hand lost — sequence discarded. Press SPACE again.")

    # Draw UI
    frame = draw_ui(frame, sequence_count, len(sequence), collecting, countdown, LABEL)

    # No hand warning
    if not hand_detected:
        cv2.putText(frame, "NO HAND DETECTED", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("SignAI — Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE → start collecting
    if key == 32 and not collecting and sequence_count < TOTAL_SEQUENCES:
        collecting = True
        countdown  = COUNTDOWN_START
        sequence   = []
        print(f"  ⏳ Get ready... recording sequence {sequence_count + 1}")

    # ESC → quit
    if key == 27 or sequence_count >= TOTAL_SEQUENCES:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n🎉 Done! Collected {sequence_count} sequences for [{LABEL}]")
print(f"📁 Saved at: {DATA_PATH}\n")

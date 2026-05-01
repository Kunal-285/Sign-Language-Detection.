import os
import time
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

# ============================================================
# CONFIG — matches your existing trained model EXACTLY
# ============================================================
LABELS = ["HELLO", "NO", "PEACE", "STOP"] # ✅ keep sorted A-Z, match collect_data.py
SEQUENCE_LENGTH      = 30       # ✅ unchanged — matches your trained model
CONFIDENCE_THRESHOLD = 0.75
HISTORY_LENGTH       = 5

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BEST_PATH  = os.path.join(BASE_DIR, "signetra_best.h5")
MODEL_PATH = os.path.join(BASE_DIR, "signetra_model.h5")
LOAD_PATH  = BEST_PATH if os.path.exists(BEST_PATH) else MODEL_PATH
print(f"✅ Loading: {os.path.basename(LOAD_PATH)}")

# ============================================================
# LOAD MODEL + WARMUP
# ============================================================
model = load_model(LOAD_PATH)
# ✅ Warmup — eliminates the lag on first real prediction
model.predict(np.zeros((1, SEQUENCE_LENGTH, 63), dtype=np.float32), verbose=0)
print("✅ Model warmed up.\n")

# ============================================================
# MEDIAPIPE — fast settings
# ============================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands            = 1,
    model_complexity         = 1,    # ✅ matches collect_data.py (must be same as training)
    min_detection_confidence = 0.6,
    min_tracking_confidence  = 0.5
)

# ============================================================
# GLOBALS
# ============================================================
sequence           = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=HISTORY_LENGTH)

# ✅ Cache last good result — no more blank flickers between frames
last_good = {"label": "", "confidence": 0, "ts": 0}

stats = {"total": 0, "success": 0, "start": time.time()}

# ============================================================
# FLASK
# ============================================================
app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)

# ============================================================
# KEYPOINT EXTRACTION
# ✅ NORMALIZED keypoints — matches exactly how collect_data.py
#    saved training data (wrist-relative, position-independent)
# ============================================================
def extract_keypoints(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark],
            dtype=np.float32
        )  # shape: (21, 3)

        # ✅ Normalize relative to wrist (landmark 0)
        # This MUST match collect_data.py → normalize_keypoints()
        wrist = raw[0]
        normalized = raw - wrist

        return normalized.flatten()  # shape: (63,)

    return None

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    stats["total"] += 1

    try:
        file = request.files.get('frame')
        if not file:
            return jsonify({'prediction': '', 'confidence': 0})

        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'prediction': '', 'confidence': 0})

        # ✅ Resize for speed
        frame = cv2.resize(frame, (320, 240))

        keypoints = extract_keypoints(frame)

        if keypoints is None:
            # Hand disappeared — clear everything for clean restart
            sequence.clear()
            prediction_history.clear()
            last_good.update({"label": "", "confidence": 0, "ts": 0})
            return jsonify({'prediction': '', 'confidence': 0})

        sequence.append(keypoints)

        # ✅ Need full 30-frame buffer before predicting
        if len(sequence) < SEQUENCE_LENGTH:
            # Tell frontend how full the buffer is (for progress display)
            return jsonify({
                'prediction' : '',
                'confidence' : 0,
                'buffer_pct' : round(len(sequence) / SEQUENCE_LENGTH * 100)
            })

        # --- Predict ---
        input_data = np.expand_dims(
            np.array(sequence, dtype=np.float32), axis=0
        )  # (1, 30, 63)

        raw_pred   = model.predict(input_data, verbose=0)[0]
        idx        = int(np.argmax(raw_pred))
        confidence = float(raw_pred[idx])

        if confidence < CONFIDENCE_THRESHOLD:
            # ✅ Return cached last good result instead of blank
            # Prevents flickering between valid predictions
            if last_good["label"] and (time.time() - last_good["ts"]) < 2.0:
                return jsonify({
                    'prediction' : last_good["label"],
                    'confidence' : last_good["confidence"],
                    'buffer_pct' : 100
                })
            return jsonify({'prediction': '', 'confidence': 0, 'buffer_pct': 100})

        label = LABELS[idx]

        # Smoothing
        prediction_history.append(label)
        final_label = max(
            set(prediction_history),
            key=list(prediction_history).count
        )

        # Cache
        last_good.update({
            "label"     : final_label,
            "confidence": round(confidence * 100, 2),
            "ts"        : time.time()
        })

        stats["success"] += 1

        return jsonify({
            'prediction' : final_label,
            'confidence' : round(confidence * 100, 2),
            'buffer_pct' : 100
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'prediction': '', 'confidence': 0})


@app.route('/health')
def health():
    return jsonify({
        'status'  : 'running',
        'model'   : os.path.basename(LOAD_PATH),
        'labels'  : LABELS,
        'uptime'  : round(time.time() - stats["start"], 1),
        'total'   : stats["total"],
        'success' : stats["success"],
        'buffer'  : len(sequence),
    })


@app.route('/reset', methods=['POST'])
def reset():
    sequence.clear()
    prediction_history.clear()
    last_good.update({"label": "", "confidence": 0, "ts": 0})
    return jsonify({'status': 'reset ok'})


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print(f"🚀 SignAI Server")
    print(f"🏷️  Labels  : {LABELS}")
    print(f"🧠 Model   : {os.path.basename(LOAD_PATH)}")
    print(f"🌐 URL     : http://127.0.0.1:5000/")
    print(f"❤️  Health  : http://127.0.0.1:5000/health")
    print("=" * 50 + "\n")
    app.run(debug=False, threaded=True)

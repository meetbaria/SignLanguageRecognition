
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from collections import Counter, deque
from pathlib import Path

# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------
MODEL_PATH = Path("models/best_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python train_model.py first."
    )

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
classes = bundle["classes"]

print("Loaded model:", bundle["model_name"])
print("Recognized classes:", classes)


# --------------------------------------------------
# 2. Extract features exactly as during training
# --------------------------------------------------
def extract_features(hand_landmarks):
    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )

    # Make coordinates relative to the wrist.
    points = points - points[0]

    # Normalize scale.
    scale = np.max(np.abs(points))

    if scale > 0:
        points = points / scale

    return points.flatten()


# --------------------------------------------------
# 3. Initialize MediaPipe and webcam
# --------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions."
    )

# Smooth predictions across recent frames.
prediction_history = deque(maxlen=7)

print("Recognition started. Press Q to quit.")

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            success, frame = cap.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb_frame)

            prediction_history.clear() if (
                not results.multi_hand_landmarks
            ) else None

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                )

                features = extract_features(hand)

                # Preserve the feature names used in training.
                input_data = pd.DataFrame(
                    [features],
                    columns=feature_columns,
                )

                probabilities = model.predict_proba(
                    input_data
                )[0]

                best_index = int(np.argmax(probabilities))
                current_letter = str(
                    model.classes_[best_index]
                )
                confidence = float(
                    probabilities[best_index]
                )

                prediction_history.append(current_letter)

                counts = Counter(prediction_history)
                stable_letter, stable_count = (
                    counts.most_common(1)[0]
                )

                # Require repeated predictions to reduce flicker.
                if stable_count >= 4:
                    display_text = (
                        f"Letter: {stable_letter}"
                    )

                    cv2.putText(
                        frame,
                        display_text,
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 255, 0),
                        3,
                    )

                    cv2.putText(
                        frame,
                        f"Current frame score: "
                        f"{confidence:.1%}",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Stabilizing prediction...",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

            else:
                cv2.putText(
                    frame,
                    "Show your hand",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Sign Language Alphabet Recognition",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

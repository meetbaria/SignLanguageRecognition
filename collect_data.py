
import csv
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# Start with 8 static ASL letters.
LETTERS = ["A", "B", "C", "D", "E", "F", "L", "Y"]

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "landmarks.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    f"f{i}" for i in range(63)
]
CSV_COLUMNS = FEATURE_COLUMNS + ["label"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_features(hand_landmarks):
    """Convert 21 landmarks into 63 normalized features."""
    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )

    # Make coordinates relative to the wrist.
    points = points - points[0]

    # Normalize scale to reduce sensitivity to hand size
    # and distance from the camera.
    scale = np.max(np.abs(points))

    if scale > 0:
        points = points / scale

    return points.flatten()


def save_sample(features, label):
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0

    with CSV_PATH.open("a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(CSV_COLUMNS)

        writer.writerow(
            [round(float(x), 6) for x in features]
            + [label]
        )


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

selected_label = None
saved_count = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
) as hands:

    print("Supported letters:", ", ".join(LETTERS))
    print("Press a letter key to select its label.")
    print("Press SPACE to save one sample.")
    print("Press Q to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        features = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            features = extract_features(hand)

        cv2.putText(
            frame,
            f"Selected label: {selected_label or 'None'}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            "Select letter, SPACE=save, Q=quit",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Collect Sign Language Data", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            if selected_label is None:
                print("Select a letter first.")
            elif features is None:
                print("No hand detected. Try again.")
            else:
                save_sample(features, selected_label)
                saved_count += 1
                print(
                    f"Saved sample {saved_count}: "
                    f"{selected_label}"
                )

        elif chr(key).upper() in LETTERS if key < 128 else False:
            selected_label = chr(key).upper()
            print(f"Selected label: {selected_label}")

cap.release()
cv2.destroyAllWindows()

print(f"Collection session finished. Samples saved: {saved_count}")
print(f"Dataset location: {CSV_PATH}")

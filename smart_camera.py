import cv2
import tensorflow as tf
import numpy as np
import os
import subprocess
from datetime import datetime

# -----------------------------
# Folders
# -----------------------------

os.makedirs("dataset/LHS", exist_ok=True)
os.makedirs("dataset/RHS", exist_ok=True)

# -----------------------------
# Counter
# -----------------------------

if not os.path.exists("counter.txt"):
    with open("counter.txt", "w") as f:
        f.write("0")

with open("counter.txt", "r") as f:
    correction_count = int(f.read().strip())

# -----------------------------
# Functions
# -----------------------------

def save_counter(count):

    with open("counter.txt", "w") as f:
        f.write(str(count))


def retrain_model():

    global model

    print("\n========================")
    print("RETRAINING MODEL")
    print("========================\n")

    subprocess.run(
        ["python3", "train.py"]
    )

    model = tf.keras.models.load_model(
        "models/lhs_rhs_model.keras"
    )

    print("\nMODEL RELOADED\n")


# -----------------------------
# Load Model
# -----------------------------

model = tf.keras.models.load_model(
    "models/lhs_rhs_model.keras"
)

# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Prediction

    img = cv2.resize(frame, (224,224))

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(
        img,
        verbose=0
    )

    lhs_prob = float(prediction[0][0]) * 100
    rhs_prob = float(prediction[0][1]) * 100

    if lhs_prob > rhs_prob:
        final_prediction = "LHS"
        confidence = lhs_prob
        color = (0,255,0)
    else:
        final_prediction = "RHS"
        confidence = rhs_prob
        color = (0,255,255)

    # UI

    cv2.rectangle(
        frame,
        (10,10),
        (550,260),
        color,
        2
    )

    cv2.putText(
        frame,
        f"LHS : {lhs_prob:.1f}%",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"RHS : {rhs_prob:.1f}%",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Prediction : {final_prediction}",
        (20,140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.putText(
        frame,
        "L=Save LHS",
        (20,190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "R=Save RHS",
        (20,220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Corrections : {correction_count}/5",
        (20,250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow(
        "Smart Wing Classifier",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # Save LHS
    # -----------------------------

    if key == ord('l'):

        filename = (
            "dataset/LHS/"
            f"lhs_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        correction_count += 1

        save_counter(correction_count)

        print(
            f"[LHS SAVED] {filename}"
        )

    # -----------------------------
    # Save RHS
    # -----------------------------

    elif key == ord('r'):

        filename = (
            "dataset/RHS/"
            f"rhs_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        correction_count += 1

        save_counter(correction_count)

        print(
            f"[RHS SAVED] {filename}"
        )

    # -----------------------------
    # Auto Retrain
    # -----------------------------

    if correction_count >= 5:

        retrain_model()

        correction_count = 0

        save_counter(correction_count)

    # Quit

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
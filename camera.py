import cv2
import tensorflow as tf
import numpy as np
import time

# Load Model
model = tf.keras.models.load_model(
    "models/lhs_rhs_model.keras"
)

# Webcam
cap = cv2.VideoCapture(0)

prev_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    # Preprocess
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(
        img,
        verbose=0
    )

    lhs_prob = float(prediction[0][0]) * 100
    rhs_prob = float(prediction[0][1]) * 100

    if lhs_prob > rhs_prob:
        final_prediction = "LHS"
        confidence = lhs_prob
        color = (0, 255, 0)
    else:
        final_prediction = "RHS"
        confidence = rhs_prob
        color = (0, 255, 255)

    # Confidence Status
    if confidence > 90:
        status = "HIGH CONFIDENCE"
    elif confidence > 70:
        status = "MEDIUM CONFIDENCE"
    else:
        status = "LOW CONFIDENCE"

    # Draw Border Box
    cv2.rectangle(
        frame,
        (10, 10),
        (420, 220),
        color,
        2
    )

    # Title
    cv2.putText(
        frame,
        "AIRCRAFT WING INSPECTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    # LHS %
    cv2.putText(
        frame,
        f"LHS : {lhs_prob:.1f}%",
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    # RHS %
    cv2.putText(
        frame,
        f"RHS : {rhs_prob:.1f}%",
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    # Prediction
    cv2.putText(
        frame,
        f"Prediction : {final_prediction}",
        (20,150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # Status
    cv2.putText(
        frame,
        status,
        (20,190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Confidence Bar Background
    cv2.rectangle(
        frame,
        (20, 240),
        (320, 270),
        (50,50,50),
        -1
    )

    # Confidence Bar
    bar_width = int((confidence / 100) * 300)

    cv2.rectangle(
        frame,
        (20,240),
        (20 + bar_width,270),
        color,
        -1
    )

    cv2.putText(
        frame,
        f"{confidence:.1f}%",
        (330,262),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time + 0.0001)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (w-140,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    # Show Camera
    cv2.imshow(
        "Aircraft Wing Classifier",
        frame
    )

    # ESC Key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
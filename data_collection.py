import cv2
import os
from datetime import datetime

# Create folders if they don't exist
os.makedirs("dataset/LHS", exist_ok=True)
os.makedirs("dataset/RHS", exist_ok=True)

cap = cv2.VideoCapture(0)

print("===================================")
print("L = Save image as LHS")
print("R = Save image as RHS")
print("Q = Quit")
print("===================================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.putText(
        frame,
        "L = Save LHS | R = Save RHS | Q = Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save LHS
    if key == ord('l'):

        filename = f"dataset/LHS/lhs_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"

        cv2.imwrite(filename, frame)

        print(f"Saved LHS: {filename}")

    # Save RHS
    elif key == ord('r'):

        filename = f"dataset/RHS/rhs_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"

        cv2.imwrite(filename, frame)

        print(f"Saved RHS: {filename}")

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
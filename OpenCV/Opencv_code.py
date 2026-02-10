import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import math

# ================== SERIAL ==================
ser = None
try:
    ser = serial.Serial(r'\\.\COM9', 115200, timeout=1)
    time.sleep(2)
    print("Serial connected on COM9")
except Exception as e:
    print("Serial not available:", e)
    print("Running OpenCV without ESP32...")

# ================== MEDIAPIPE ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ================== OPENCV ==================
cap = cv2.VideoCapture(0)

# Previous angles for smoothing
prev_angles = [0, 0, 0, 0, 0]
SMOOTHING = 0.5

def dist_angle(tip, base, min_d, max_d):
    d = np.linalg.norm(np.array(tip) - np.array(base))
    return int(np.clip(np.interp(d, [min_d, max_d], [0, 180]), 0, 180))

def thumb_angle(lm):
    """
    High-accuracy thumb angle using vector geometry
    (direction FIXED)
    """
    wrist = np.array(lm[0])
    mcp   = np.array(lm[2])
    tip   = np.array(lm[4])

    v1 = wrist - mcp
    v2 = tip - mcp

    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    # 🔥 FIX: invert thumb direction
    # OPEN ≈ 160°, CLOSED ≈ 40°
    servo = np.interp(angle, [40, 160], [0, 180])

    return int(np.clip(servo, 0, 180))

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    angles = prev_angles.copy()

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        lm = [(int(p.x * w), int(p.y * h)) for p in hand.landmark]

        # ✅ Thumb (vector-based, inverted correctly)
        angles[0] = thumb_angle(lm)

        # Index (inverted)
        angles[1] = 180 - dist_angle(lm[8], lm[5], 20, 100)

        # Middle
        angles[2] = dist_angle(lm[12], lm[9], 20, 110)

        # Ring
        angles[3] = dist_angle(lm[16], lm[13], 20, 105)

        # Pinky
        angles[4] = dist_angle(lm[20], lm[17], 15, 95)

        # ================== SMOOTHING ==================
        for i in range(5):
            angles[i] = int(
                SMOOTHING * prev_angles[i] +
                (1 - SMOOTHING) * angles[i]
            )

        prev_angles = angles.copy()

        data = ",".join(map(str, angles)) + "\n"
        if ser and ser.is_open:
            ser.write(data.encode())

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        cv2.putText(
            frame,
            data,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================== CLEANUP ==================
cap.release()
if ser and ser.is_open:
    ser.close()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# ================== SERIAL ==================
ser = serial.Serial('COM3', 115200, timeout=1)  # change COM port
time.sleep(2)

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

FINGERS = [
    (4, 2),    # Thumb
    (8, 5),    # Index
    (12, 9),   # Middle
    (16, 13),  # Ring
    (20, 17)   # Pinky
]

def finger_angle(tip, base):
    dist = np.linalg.norm(np.array(tip) - np.array(base))
    angle = np.interp(dist, [20, 100], [0, 180])
    return int(np.clip(angle, 0, 180))

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    angles = [0]*5

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        lm = [(int(p.x*w), int(p.y*h)) for p in hand.landmark]

        for i, (tip, base) in enumerate(FINGERS):
            angles[i] = finger_angle(lm[tip], lm[base])

        data = ",".join(map(str, angles)) + "\n"
        ser.write(data.encode())

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, data, (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
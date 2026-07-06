import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ---------------- SETUP ----------------
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

last_click_time = 0
click_delay = 1  # seconds

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            middle_x = int(middle_tip.x * frame_width)
            middle_y = int(middle_tip.y * frame_height)

            # Draw line (Middle -> Thumb)
            cv2.line(
                frame,
                (middle_x, middle_y),
                (thumb_x, thumb_y),
                (255, 0, 0),
                3
            )

            # Distance for right click
            right_click_distance = math.hypot(
                middle_x - thumb_x,
                middle_y - thumb_y
            )

            # ---------------- RIGHT CLICK ----------------
            if right_click_distance < 40:

                if current_time - last_click_time > click_delay:

                    pyautogui.rightClick()

                    last_click_time = current_time

                    cv2.putText(
                        frame,
                        "RIGHT CLICK",
                        (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3
                    )

    # ---------------- SHOW WINDOW ----------------
    cv2.imshow("AI Virtual Mouse", frame)

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
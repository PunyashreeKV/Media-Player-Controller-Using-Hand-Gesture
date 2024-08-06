import cv2
import numpy as np
import mediapipe as mp
import pyautogui as p

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Read Camera
cap = cv2.VideoCapture(0)

# Initial state
is_playing = False
hand_present = False

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (600, 500))
    
    # Convert image to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process image with MediaPipe
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmark coordinates
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Gesture recognition
            def recognize_gesture():
                # Thumb up for volume up
                if thumb_tip.y < thumb_ip.y and all(finger_tip.y > thumb_tip.y for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
                    return "volume_up"
                # Any two fingers up for volume down
                fingers_up = sum(finger_tip.y < thumb_tip.y for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip])
                if fingers_up == 2:
                    return "volume_down"
                return None

            gesture = recognize_gesture()
            if gesture:
                print("Gesture:", gesture)
                if gesture == "volume_up":
                    p.press('volumeup')
                    cv2.putText(frame, "Volume UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif gesture == "volume_down":
                    p.press('volumedown')
                    cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        # Toggle play/pause if hand is present and was previously not detected
        if not hand_present:
            p.press('playpause')
            is_playing = not is_playing
            hand_present = True
            cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    else:
        hand_present = False

    cv2.imshow("Hand Gesture Media Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
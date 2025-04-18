import cv2
import mediapipe as mp
import os


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# File to store dataset
DATA_FILE = "hand_gestures.csv"

#  it will check if the csv file exis or no
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        f.write("label," + ",".join([f"x{i},y{i},z{i}" for i in range(21)]) + "\n")

cap = cv2.VideoCapture(0)
current_label = None  # Store the current label

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display current label on the screen
    if current_label:
        cv2.putText(frame, f"Label: {current_label}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking - 'L' to Set Label, 'S' to Save", frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('l'):  # Press 'L' to set the label once
        current_label = input("Enter gesture label: ")
        print(f"Label set to: {current_label}")

    elif key == ord('s') and current_label:  # Press 'S' to save data
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = [current_label]  # Store label
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])  # Store XYZ coords
                
                # Save to CSV
                with open(DATA_FILE, "a") as f:
                    f.write(",".join(map(str, coords)) + "\n")
                print(f"Saved: {current_label}")

    elif key == ord('q'):  # Press 'Q' to quit
        break

cap.release()
cv2.destroyAllWindows()

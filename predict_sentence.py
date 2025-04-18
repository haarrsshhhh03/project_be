import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model---random forest
# model = joblib.load("model.pkl")  

#svm
model = joblib.load("svm_model.pkl")


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sentence = []  # for storing the words
current_word = []  # Stores letters forming a word

print("\n Camera is ON. Press ENTER to capture sign. SPACE for new word. Type 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(" Camera error!")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show camera feed
    cv2.putText(frame, "ENTER = Capture, SPACE = New Word, 'q' = Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Detection", frame)

    key = cv2.waitKey(1) & 0xFF  
    if key == ord('q'):  
        break
    elif key == 13:  # ENTER key
        print("\n Capturing sign...")

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])

                coords = np.array(coords).reshape(1, -1)

                # Predict the letter
                prediction = model.predict(coords)[0]

                if prediction == "SPACE":
                    if current_word:
                        sentence.append("".join(current_word))  # Add full word to sentence
                        current_word = []  # Reset word
                else:
                    current_word.append(prediction)  # Add letter to current word

        # Print updated sentence in terminal
        print("\n Current Sentence:", " ".join(sentence + ["".join(current_word)]))

    elif key == 32:  # SPACE key (new word)
        if current_word:
            sentence.append("".join(current_word))  # Add full word to sentence
            current_word = []  # Reset word
        print("\n New Word Started!")

cap.release()
cv2.destroyAllWindows()

# Final sentence
final_sentence = " ".join(sentence + ["".join(current_word)])
print("\n Final Sentence:", final_sentence)

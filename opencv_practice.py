import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("Error: Camera not detected.")
    exit()

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Flip image horizontally to match mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert BGR to RGB (for MediaPipe if needed)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # Apply Edge Detection (Canny)
    edges = cv2.Canny(gray, 100, 200)

    # Draw a rectangle on screen
    cv2.rectangle(frame, (50, 50), (250, 250), (0, 255, 0), 2)

    # Draw a circle on screen
    cv2.circle(frame, (320, 240), 50, (0, 0, 255), -1)

    # Draw a line on screen
    cv2.line(frame, (100, 100), (400, 400), (255, 0, 0), 3)

    # Add text to screen
    cv2.putText(frame, "Hello harsh how are you", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display all processed frames
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edge Detection", edges)

    # Press 's' to save image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("saved_frame.jpg", frame)
        print("Image saved as 'saved_frame.jpg'")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

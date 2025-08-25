import cv2
from deepface import DeepFace

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

frame_count = 0
last_emotion = "Detecting..."

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    frame_count += 1

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle always
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Only run DeepFace every 5 frames
        if frame_count % 5 == 0:
            face_roi = frame[y:y + h, x:x + w]

            # Resize face for faster inference
            face_roi = cv2.resize(face_roi, (224, 224))

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                last_emotion = result[0]['dominant_emotion']
            except Exception as e:
                print(f"Error in emotion detection: {e}")
                last_emotion = "Error"

        # Always display the last known emotion
        cv2.putText(frame, last_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame with the emotion label
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit conditions
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Real-time Emotion Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

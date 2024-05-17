import cv2
import sqlite3

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/trainer.yml')

# Load the face cascade
face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')

# Initialize and start the video capture
cap = cv2.VideoCapture(0)

# Connect to the database to fetch person names
connection = sqlite3.connect('Person.db')
cursor = connection.cursor()

# Font for displaying text on the video
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Recognize the face
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Check if confidence is less than 100 --> "0" is perfect match
        if confidence < 100:
            cursor.execute("SELECT Name FROM Persons WHERE Id = ?", (id,))
            result = cursor.fetchone()
            name = result[0] if result else "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"
        
        # Display the name and confidence on the frame
        cv2.putText(frame, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    # Display the frame with the recognized face
    cv2.imshow('Face Recognition', frame)

    # Break the loop if the 'ESC' key is pressed
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release the video capture and close the database connection
cap.release()
connection.close()
cv2.destroyAllWindows()

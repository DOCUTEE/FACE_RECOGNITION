import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import os
import time
import sqlite3
import numpy as np
from PIL import Image

# Global variables
face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create the main window
window = tk.Tk()
window.title("Face Recognition App")

def create_data():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)
    img_counter = 0

    # Ask user for the name of the person in the video
    name = simpledialog.askstring("Input", "Enter the name of the person:")

    if not name:
        messagebox.showwarning("Input Error", "Name cannot be empty!")
        return

    connection = sqlite3.connect('Person.db')
    cursor = connection.cursor()
    # Check if the table exists
    cursor.execute("PRAGMA table_info('Persons')")
    table_exists = cursor.fetchall()

    # If the table does not exist, create it
    if not table_exists:
        cursor.execute('''CREATE TABLE Persons(
            Id INTEGER,
            Name TEXT
        )''')
        connection.commit()
        print("Table 'Persons' created.")
    else:
        print("Table 'Persons' already exists.")
    cursor.execute("SELECT MAX(Id) FROM Persons")
    max_id = cursor.fetchone()[0]
    if max_id is None:
        max_id = 0
    max_id += 1

    # The directory where we will store the images
    dir_name = "dataset/"

    # Create directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    while img_counter < 30:
        # Read the frame
        ret, img = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw the rectangle around each face and save the image
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            img_name = dir_name + "/{0}_{1}.png".format(max_id, time.time())
            cv2.imwrite(img_name, roi_gray)
            img_counter += 1

        # Display the video frame with rectangles
        cv2.imshow('Creating Data', img)

        # Stop if escape key is pressed
        if cv2.waitKey(30) & 0xFF == 27:
            break

        # Wait for 0.1 seconds
        time.sleep(0.1)

    if img_counter > 0:
        sql_str = "INSERT INTO Persons (Id, Name) VALUES (?, ?)"
        cursor.execute(sql_str, (max_id, name))
        connection.commit()

    # Release the VideoCapture object and close the database connection
    connection.close()
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Data creation completed successfully!")

    # Train the recognizer with the new data
    train_recognizer()

def train_recognizer():
    # Path for face image database
    path = 'dataset'

    # Function to get the images and label data
    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
        face_samples = []
        ids = []

        for image_path in image_paths:
            pil_image = Image.open(image_path).convert('L')  # convert it to grayscale
            img_numpy = np.array(pil_image, 'uint8')

            # Extract the person's id from the image name
            id = int(os.path.split(image_path)[-1].split('_')[0])
            
            faces = face_cascade.detectMultiScale(img_numpy)
            
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        
        return face_samples, ids

    print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into model/trainer.yml
    if not os.path.exists('model'):
        os.makedirs('model')
    recognizer.save('model/trainer.yml')

    # Print the number of faces trained and end program
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting Program.")
    messagebox.showinfo("Info", "Training completed successfully!")

def face_recognition():
    recognizer.read('model/trainer.yml')

    # Connect to the database to fetch person names
    connection = sqlite3.connect('Person.db')
    cursor = connection.cursor()

    # Initialize and start the video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Recognize the face
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

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
            cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        # Display the frame with the recognized face
        cv2.imshow('Face Recognition', frame)

        # Break the loop if the 'ESC' key is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # Release the video capture and close the database connection
    cap.release()
    connection.close()
    cv2.destroyAllWindows()

# Create buttons for adding data and recognizing faces
btn_create_data = tk.Button(window, text="Add Face Data", command=create_data)
btn_create_data.pack(pady=10)

btn_face_recognition = tk.Button(window, text="Face Recognition", command=face_recognition)
btn_face_recognition.pack(pady=10)

# Start the GUI main loop
window.mainloop()

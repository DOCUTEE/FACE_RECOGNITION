import cv2
import os
import time
import sqlite3

# Load the cascade
face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

# Counter for the number of images taken
img_counter = 0

# Ask user for the name of the person in the video
name = input("Enter the name of the person in the video: ")

connection = sqlite3.connect('Person.db')
cursor = connection.cursor()
cursor.execute("SELECT MAX(Id) FROM Persons")
max_id = cursor.fetchone()[0]
max_id += 1



# The directory where we will store the images
dir_name = "dataset/"

# Create directory if it doesn't exist
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

while img_counter < 30:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw the rectangle around each face and save the image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        img_name = dir_name + "/{0}_{1}.png".format(max_id,time.time())
        cv2.imwrite(img_name, roi_gray)
        img_counter += 1

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
     # Wait for 0.1 seconds
    time.sleep(0.1)


if (img_counter > 0):
    sqlStr = "INSERT INTO Persons VALUES ({0},'{1}')".format(max_id,name)
    cursor.execute(sqlStr)
# Release the VideoCapture object
connection.commit()
connection.close()
cap.release()

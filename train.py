import cv2
import os
import numpy as np
from PIL import Image

# Path for face image database
path = 'dataset'

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haar_cascade_files/haarcascade_frontalface_default.xml")

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
        
        faces = detector.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    
    return face_samples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

# Save the model into model/trainer.yml
if not os.path.exists('model'):
    os.makedirs('model')
recognizer.save('model/trainer.yml')

# Print the number of faces trained and end program
print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program.")

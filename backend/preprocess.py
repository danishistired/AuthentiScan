import cv2
import os
import numpy as np
from mtcnn import MTCNN

# Initialize MTCNN for face detection
detector = MTCNN()

def extract_faces(video_path, output_folder, frame_interval=10):
    """
    Extract faces from a video at regular intervals.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frames at regular intervals
        if frame_count % frame_interval == 0:
            # Detect faces in the frame
            faces = detector.detect_faces(frame)
            for i, face in enumerate(faces):
                x, y, width, height = face['box']
                face_img = frame[y:y+height, x:x+width]
                face_img = cv2.resize(face_img, (224, 224))  # Resize to 224x224
                cv2.imwrite(os.path.join(output_folder, f"face_{face_count}.jpg"), face_img)
                face_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {face_count} faces from {video_path}")

# Example usage
extract_faces("../sample_video.mp4", "output_faces")
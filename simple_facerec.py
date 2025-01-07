import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        # Resize frame for a faster speed
        self.frame_resizing = 0.50

    def load_encoding_images(self, images_path):
        """
        Load encoding images from the specified path.
        :param images_path: Path to the folder containing images.
        """
        # Load all images from the folder
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            try:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Get the filename without the extension
                basename = os.path.basename(img_path)
                filename, ext = os.path.splitext(basename)
                # Encode the face in the image
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                # Store the encoding and name
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
            except IndexError:
                print(f"Warning: No face detected in image {img_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        print("Encoding images loaded.")

    def detect_known_faces(self, frame):
        """
        Detect and recognize known faces in a given video frame.
        :param frame: A single frame from the video.
        :return: Face locations and face names.
        """
        if frame is None:
            print("Error: Frame is None.")
            return [], []

        try:
            # Resize the frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
            # Convert the frame from BGR to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            # Detect face locations and encodings
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Compare detected faces with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                # Find the known face with the smallest distance
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)

            # Convert face locations back to the original frame size
            face_locations = np.array(face_locations) / self.frame_resizing
            return face_locations.astype(int), face_names
        except Exception as e:
            print(f"Error during face detection: {e}")
            return [], []


# Example Usage:
# Ensure to validate your main_video.py to handle errors while reading frames.
if __name__ == "__main__":
    sfr = SimpleFacerec()
    sfr.load_encoding_images("path_to_images_folder")
    cap = cv2.VideoCapture(0)  # Replace 0 with the path to your video file if necessary

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read video frame.")
            break

        face_locations, face_names = sfr.detect_known_faces(frame)

        # Display results
        for face_loc, name in zip(face_locations, face_names):
            top, right, bottom, left = face_loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

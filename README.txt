Title : Smart face based access control

Problem Statement:
Managing and monitoring access control in real-time environments often requires manual effort and lacks automation, accuracy, and accountability.

What Our Project Overcomes:
This project automates face recognition-based access control, ensuring real-time detection, logging of entry and exit times, and accurate tracking of authorized and unauthorized individuals.


Abstract View of Two Key Codes:
Face Encoding and Detection: The first part encodes images from a predefined folder to identify known faces by generating unique face encodings and storing their names.
Real-Time Recognition and Logging: The second part uses these encodings to recognize faces in live video, log entry and exit times in a CSV file, and track durations while evaluating system performance.

Modules Used:
face_recognition: For face detection and encoding.
cv2 (OpenCV): For real-time video capture and image processing.
numpy: For mathematical operations and array manipulations.
pandas: For reading and updating the CSV log file.
time and os: For time tracking and file operations.

video_demo:
https://drive.google.com/file/d/1YI-IT371lJqWzzLw1--HWZIDQPw4czFC/view?usp=sharing
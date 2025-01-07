import cv2

cap = cv2.VideoCapture(0)  # Use appropriate index or path
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Test Frame", frame)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

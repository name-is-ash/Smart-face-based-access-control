import cv2
from simple_facerec import SimpleFacerec
import csv
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

# Log file setup
log_file = "detected_faces_log.csv"
 
# Check if log file exists; if not, create it with appropriate headers
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "In-Time", "Out-Time", "Duration", "Authorized"])

# Function to log the face
def log_face(name, in_out, is_authorized):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Format time with seconds
    authorized_status = "Authorized" if is_authorized else "Unauthorized"
    
    try:
        df = pd.read_csv(log_file)
    except pd.errors.ParserError:
        print(f"Error reading {log_file}. There might be corrupted data.")
        return

    if in_out == "IN":
        # Create a new row for "IN" time
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, current_time, "", "", authorized_status])  # Empty "Out-Time" and "Duration"
    
    elif in_out == "OUT":
        # Find the last "IN" entry for this person without an "OUT" time
        last_in_entry = df[(df["Name"] == name) & (df["Out-Time"].isna())]
        
        if not last_in_entry.empty:
            # Update the "Out-Time" for the last "IN" entry
            index = last_in_entry.index[-1]
            
            # Ensure that the "Out-Time" column can accept strings (for timestamps)
            df["Out-Time"] = df["Out-Time"].astype(str)

            df.loc[index, "Out-Time"] = current_time
            
            # Calculate duration between "In-Time" and "Out-Time"
            in_time_str = df.loc[index, "In-Time"]
            in_time = pd.to_datetime(in_time_str)
            out_time = pd.to_datetime(current_time)
            duration = (out_time - in_time).total_seconds()
            
            # Update the "Duration" column with the calculated time difference
            df.loc[index, "Duration"] = duration
            
            # Save the updated dataframe to the CSV file
            df.to_csv(log_file, index=False)
            print(f"{name} - Duration calculated: {duration} seconds.")

# Dictionary to track the last time a face was detected
last_detected = {}

# Timer and cooldown period
COOLDOWN_PERIOD = 120  # 2 minutes

# Lists for evaluation
true_labels = []  # Ground truth
predicted_labels = []  # Model predictions

while True:
    
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Frame is None.")
        continue


    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    # For each detected face
    if face_names:
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

            current_time = time.time()

            # Mark as "Authorized" if face is known, else "Unauthorized"
            is_authorized = name != "Unknown"

            # If the face is detected for the first time or after the cooldown period
            if name not in last_detected or (current_time - last_detected[name]) > COOLDOWN_PERIOD:
                log_face(name, "IN", is_authorized)
                last_detected[name] = current_time  # Update the time of last detection
            else:
                # If the face is detected again and it's been at least 10 seconds, log "OUT"
                if (current_time - last_detected[name]) >= 10:  # 10-second gap
                    log_face(name, "OUT", is_authorized)
                    last_detected[name] = current_time  # Reset time after logging "OUT"

            # Here we assume the true label is the name of the detected face (as per your testing setup)
            true_label = name  # In real scenarios, get the true label from ground truth data
            true_labels.append(true_label)
            predicted_labels.append(name)

    # Display the frame with the recognized face
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  
        break

cap.release()
cv2.destroyAllWindows()

# Evaluation Metrics Calculation
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
cm = confusion_matrix(true_labels, predicted_labels)

# Print Metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("Confusion Matrix:")
print(cm)

# Evaluate performance for known and unknown faces
known_faces_accuracy = accuracy_score([t for t in true_labels if t != 'Unknown'], 
                                      [p for p in predicted_labels if p != 'Unknown'])
unknown_faces_detection_rate = recall_score([t == 'Unknown' for t in true_labels], 
                                            [p == 'Unknown' for p in predicted_labels])

print(f"Known Faces Accuracy: {known_faces_accuracy:.2f}")
print(f"Unknown Faces Detection Rate: {unknown_faces_detection_rate:.2f}")

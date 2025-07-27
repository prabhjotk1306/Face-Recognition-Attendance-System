import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

print("Webcam Connection established")

# Load images and class names
images = []
classNames = []
path = r"photos"

if not os.path.exists(path):
    print(f"[ERROR] Directory '{path}' not found.")
    exit()

myList = os.listdir(path)
print(f"[INFO] Found {len(myList)} files in '{path}'")

for cl in myList:
    full_path = os.path.join(path, cl)
    curImg = cv2.imread(full_path)

    if curImg is None:
        print(f"[WARNING] Unable to load image: {full_path}")
        continue

    if curImg.dtype != np.uint8:
        print(f"[WARNING] Image not 8-bit, fixing: {full_path}")
        curImg = curImg.astype(np.uint8)

    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0].strip().lower())

print(f"[INFO] Loaded {len(images)} valid images.")

# Load student details from CSV
student_details = {}
try:
    with open('input.csv', mode='r') as file:
        csvReader = csv.DictReader(file)
        for row in csvReader:
            name_key = row['Name'].strip().lower()
            student_details[name_key] = row
    print(f"[INFO] Loaded {len(student_details)} student records from CSV.")
except FileNotFoundError:
    print("[ERROR] input.csv not found.")
    exit()

def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        print(f"[DEBUG] Encoding image {idx+1}/{len(images)}")
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ERROR] Failed to convert image {idx+1} to RGB: {e}")
            continue

        if img_rgb is None or img_rgb.dtype != np.uint8:
            print(f"[ERROR] Invalid image passed for encoding at index {idx}")
            continue

        try:
            encodings = face_recognition.face_encodings(img_rgb)
            if not encodings:
                print(f"[WARNING] No face found in image {idx+1}")
                continue
            encodeList.append(encodings[0])
        except Exception as e:
            print(f"[ERROR] face_recognition failed at image {idx+1}: {e}")
    return encodeList

def markAttendance(name, student_id):
    try:
        with open('Attendencebook.csv', 'r+', newline='') as f:
            myDataList = f.readlines()
            nameList = [line.split(',')[0] for line in myDataList]

            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%Y-%m-%d %H:%M:%S')
                f.writelines(f'\n{name},{student_id},{dtString}')
    except FileNotFoundError:
        print("[ERROR] Attendencebook.csv not found. Creating one...")
        with open('Attendencebook.csv', 'w', newline='') as f:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write("Name,Student Id,Timestamp")
            f.writelines(f'\n{name},{student_id},{dtString}')

encodeListKnown = findEncodings(images)
print(f"[INFO] Encoding of {len(encodeListKnown)} images completed.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

try:
    while True:
        success, img = cap.read()
        if not success or img is None:
            print("[ERROR] Failed to read from webcam.")
            break

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        try:
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ERROR] Failed to convert webcam frame to RGB: {e}")
            continue

        try:
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        except Exception as e:
            print(f"[ERROR] face_recognition processing error: {e}")
            continue

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].strip().lower()
                details = student_details.get(name)
                if details:
                    display_text = f"ID: {details['Student Id']}, Name: {details['Name']}, Branch: {details['Branch']}"
                    student_id = details['Student Id']
                else:
                    print(f"[WARNING] No CSV details for recognized name: {name}")
                    display_text = "Details not found"
                    student_id = 'Unknown'

                y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (10, 10), (650, 60), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, display_text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                markAttendance(name, student_id)
            else:
                name = 'Not Found'
                y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting loop on user request.")
            break
finally:
    print("[INFO] Releasing camera and closing windows.")
    cap.release()
    cv2.destroyAllWindows()

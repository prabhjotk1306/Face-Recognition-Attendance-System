
# Face Recognition Attendance System 

This is a Python-based Face Recognition Attendance System using `face_recognition`, `OpenCV`. It captures real-time webcam input to detect and identify known faces and logs their attendance with a timestamp.

---

## 📁 Project Structure

```

Face\_rec\_attendence/
├── photos/                  # Folder with known face images 
├── Att.py                   # Face recognition and attendance logic                 
├── input.csv                # Input data
├── Attandencebook.csv       # Logged attendance records
├── requirements.txt         # Python dependencies
└── README.md                # This file

````

---

## Technologies Used

- **Python 3.10+**
- **OpenCV** – for real-time webcam capture
- **face_recognition** – built on top of `dlib` for facial recognition
- **dlib** – performs face encodings & comparisons
- **NumPy** – image array processing


---

##  Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Face_rec_attendence.git
cd Face_rec_attendence
````

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate    
```

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

> ⚠️ **Note on dlib:**
>  Used pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl as installing directly was giving problems

---

## 📸 How It Works

1. Place known faces in the `photos/` folder (e.g., `Modi.jpg`).
2. The system loads these images, encodes them using `dlib`.
3. When you run `Att.py`, it opens the webcam and checks every frame for faces.
4. If a face is recognized, their name and current time are recorded in `Attandencebook.csv`.

---

## ▶️ Running the App

```bash
python Att.py
```

You should see a webcam window with detected faces and names. The script auto-logs their attendance.

---

## ✅ Sample Output

`Attandencebook.csv`:

```
Name,Time
Modi,2025-07-27 20:43:15

```

---

## 📌 Features

* Real-time webcam face recognition
* Easy to add/remove known faces
* Attendance saved automatically
* Uses `dlib` under the hood for highly accurate encodings
* Easy to expand with GUI or liveness detection








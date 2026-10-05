# 😴 Drowsiness Detection System

## 📌 Project Overview

The Drowsiness Detection System is a computer vision project designed to identify signs of driver fatigue by analyzing facial landmarks and eye movements. It uses Python, OpenCV, and dlib to detect faces and facial landmarks, helping identify potential drowsiness.

The project demonstrates how computer vision can be applied to driver safety and fatigue monitoring.

## 🎯 Objectives

- Detect human faces using dlib.
- Identify facial landmarks using a 68-point facial landmark predictor.
- Analyze eye movements to help detect drowsiness.
- Provide a foundation for a driver alert system.
- Apply computer vision techniques to a real-world safety problem.

## ✨ Features

- Face detection using dlib's frontal face detector.
- Facial landmark detection using a pretrained 68-point predictor.
- Eye-region analysis for drowsiness detection.
- Webcam-based monitoring, if enabled in the implementation.
- Visual feedback when drowsiness is detected, if implemented.

## 🛠️ Technologies Used

- Python
- dlib
- OpenCV
- NumPy
- imutils
- SciPy
- Streamlit (for the web interface, if used)

## 📂 Project Structure

```text
drowsiness-detection/
│
├── app.py
├── requirements.txt
├── shape_predictor_68_face_landmarks.dat
└── README.md
```

**Files:**
- `app.py` — Main Python application. Replace this name if your entry file has a different name.
- `requirements.txt` — Lists the required Python packages.
- `shape_predictor_68_face_landmarks.dat` — Pretrained model used to locate facial landmarks.
- `README.md` — Project documentation.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd drowsiness-detection
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

If your project uses Streamlit:

```bash
streamlit run app.py
```

If it is a standard Python application:

```bash
python app.py
```

## 🧠 How It Works

1. The application obtains frames from the selected image or video source.
2. dlib detects faces in the frame.
3. The facial landmark predictor identifies 68 facial points.
4. The eye landmarks can be used to analyze eye closure and calculate the Eye Aspect Ratio (EAR), if implemented.
5. The application uses the implemented detection logic to determine whether signs of drowsiness are present.
6. The result can be displayed visually or through an alert, depending on the implementation.

## 📐 Eye Aspect Ratio (EAR)

The Eye Aspect Ratio is a common computer vision technique for measuring eye openness using facial landmarks. A decrease in EAR over a period of time can indicate prolonged eye closure.

The result depends on the detection threshold, duration, and implementation. EAR alone is not a definitive measure of driver fatigue.

## 📦 Requirements

A starting `requirements.txt` for a Streamlit-based project is:

```text
streamlit
dlib
opencv-python-headless
numpy
imutils
scipy
Pillow
```

Keep only the packages your code uses, and verify compatible versions during deployment.

## ☁️ Deployment

The project can be adapted for deployment using Streamlit Community Cloud.

1. Upload the project files to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select the repository and branch.
5. Set the main file path to your Streamlit entry file.
6. Choose Python 3.12 as the initial compatibility target.
7. Deploy the application and review build logs if errors occur.

**Note:** Cloud servers cannot directly access your personal computer's webcam through `cv2.VideoCapture(0)`. A browser-based camera interface is needed for webcam input from website visitors. Also, ensure the facial landmark model file is available to the deployed application.

## 🚀 Future Improvements

- Improve drowsiness detection accuracy.
- Add configurable EAR thresholds and duration checks.
- Add audio alerts for prolonged eye closure.
- Display detection statistics.
- Improve the interface and visualization.
- Evaluate performance under different lighting conditions.

## ⚠️ Limitations

- Detection accuracy depends on lighting, camera quality, face position, and model performance.
- Facial landmarks may be inaccurate when the face is partially obscured.
- The application is a demonstration project and should not be relied upon as the sole driver-safety mechanism.

## 👨‍💻 Author

**Deep Menpara**

B.Tech in Information Technology

GitHub: [Deep7133](https://github.com/Deep7133)

## 📄 License

This project is intended for educational and demonstration purposes. Add an appropriate open-source license if you want others to reuse, modify, or distribute the project.

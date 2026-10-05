# 😴 Drowsiness Detection System

## 📌 Project Overview

The **Drowsiness Detection System** is a computer vision application developed using Python, OpenCV, dlib, and Streamlit. It monitors eye activity by detecting facial landmarks and calculating the Eye Aspect Ratio (EAR) to identify signs of drowsiness.

When the eyes remain closed for a specified number of consecutive frames, the system triggers an alert to help indicate potential drowsiness.

The project demonstrates the practical application of computer vision and facial landmark detection for driver safety and fatigue monitoring.

## 🎯 Objectives

- Detect human faces using dlib.
- Identify facial landmarks using a pretrained 68-point facial landmark model.
- Calculate the Eye Aspect Ratio (EAR).
- Detect prolonged eye closure.
- Generate an alert when drowsiness is detected.
- Support multiple input sources, including webcam, images, videos, and URL/IP camera streams.
- Provide an interactive interface using Streamlit.

## ✨ Features

- **Real-Time Drowsiness Detection:** Monitor eye activity through a webcam.
- **Facial Landmark Detection:** Detect 68 facial landmarks using dlib.
- **Eye Aspect Ratio (EAR):** Analyze eye openness to help identify closed eyes.
- **Alert System:** Generate an alert when eye closure exceeds a configured frame threshold.
- **Image Detection:** Upload an image and analyze facial landmarks.
- **Video Detection:** Process a video file for drowsiness detection.
- **URL/IP Camera Detection:** Accept a supported camera stream URL.
- **Streamlit Interface:** Interact with the system through a web application.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image and video processing |
| dlib | Face detection and facial landmark prediction |
| Streamlit | Web application interface |
| NumPy | Numerical and array operations |
| SciPy | Distance calculations for EAR |
| imutils | Image-processing utilities |
| winsound | Alert beep on supported Windows systems |

## 📂 Project Structure

```text
drowsiness-detection/
│
├── app.py
├── requirements.txt
├── shape_predictor_68_face_landmarks.dat
└── README.md
```

**File descriptions**

- `app.py` — Main Streamlit application containing the detection logic.
- `requirements.txt` — Lists the Python dependencies.
- `shape_predictor_68_face_landmarks.dat` — Pretrained facial landmark model used to identify facial points.
- `README.md` — Project documentation.

If your main Python file has a different name, replace `app.py` with your actual filename.

## 📦 Prerequisites

- Python 3.12 as the initial compatibility target.
- pip package manager.
- A webcam for live detection, or an image/video file for offline detection.
- The pretrained dlib facial landmark model.

### Download the facial landmark model

The project requires the following model file:

`shape_predictor_68_face_landmarks.dat`

Download it from the official dlib model archive:

[Download the 68-point facial landmark predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

After downloading:

1. Extract the `.bz2` archive.
2. Place `shape_predictor_68_face_landmarks.dat` in the project directory.
3. Ensure that the filename matches the one used in your Python code.

## ⚙️ Installation and Setup

### 1. Clone the repository

Replace the example URL with your actual GitHub repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd drowsiness-detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install the dependencies

Create a `requirements.txt` file containing:

```text
streamlit
opencv-python
dlib
imutils
scipy
numpy
```

Install the packages:

```bash
pip install -r requirements.txt
```

For a headless Linux deployment, use `opencv-python-headless` instead of `opencv-python`. Install only one OpenCV package variant in the same environment.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## 🧠 Project Workflow

1. **Input:** Obtain frames from a webcam, image, video, or supported camera URL.
2. **Face Detection:** Detect the face using dlib's frontal face detector.
3. **Facial Landmark Detection:** Locate facial points using the 68-point predictor.
4. **Eye Landmark Extraction:** Identify the points surrounding each eye.
5. **EAR Calculation:** Calculate the Eye Aspect Ratio to estimate eye openness.
6. **Drowsiness Detection:** Compare EAR with a configured threshold and monitor consecutive frames.
7. **Alert Generation:** Trigger an alert when prolonged eye closure meets the configured condition.
8. **Visualization:** Display the detection result in the application.

## 📐 Eye Aspect Ratio (EAR)

The Eye Aspect Ratio is a measurement derived from the distances between facial landmarks around the eye.

It helps estimate whether the eyes are open or closed.

- **Eyes open:** The vertical distances between eye landmarks are relatively large.
- **Eyes closed:** The vertical distances decrease, resulting in a lower EAR value.
- **Prolonged eye closure:** If EAR stays below the configured threshold for enough consecutive frames, the system identifies a potential drowsiness event.

The system uses a frame counter to reduce false alerts caused by brief blinks. The EAR threshold and frame limit depend on the implementation and configuration.

## 🚨 Alert System

The alert system monitors the duration of eye closure.

1. When EAR falls below the configured threshold, the frame counter increases.
2. When EAR returns to the normal range, the counter resets.
3. When the counter exceeds the configured frame limit, the system triggers an alert.

The project report specifies `winsound` for the beep alert. Because `winsound` is Windows-specific, a different sound implementation may be required on Linux-based cloud platforms.

## ☁️ Deployment on Streamlit Community Cloud

The application can be adapted for deployment using Streamlit Community Cloud.

1. Upload the source code and required files to GitHub.
2. Ensure that `requirements.txt` lists the dependencies.
3. Ensure that the facial landmark model file is accessible to the application.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/).
5. Sign in with GitHub and create a new app.
6. Select the repository, branch, and main Python file.
7. Select a compatible Python version and deploy.
8. Review the build logs if a dependency fails to install.

### Important deployment notes

- Cloud servers cannot directly access your personal computer's webcam through `cv2.VideoCapture(0)`.
- Use a browser-based camera interface if website visitors need to provide webcam input.
- A continuous real-time webcam experience may require changes to the original application.
- Use `opencv-python-headless` for a headless Linux environment.
- Replace Windows-only sound functionality if deploying on Linux.
- Ensure all required model files are available in the deployed environment.

## ✅ Advantages

- Uses facial landmarks to analyze eye activity.
- Supports multiple input types.
- Demonstrates real-time computer vision techniques.
- Uses a pretrained landmark model rather than training a facial landmark detector from scratch.
- Provides an interactive interface through Streamlit.
- Can be extended with additional monitoring and alert features.

## ⚠️ Limitations

- Detection accuracy may decrease in poor lighting.
- Glasses and partially obscured eyes can affect landmark detection.
- Extreme face angles may reduce detection quality.
- URL/IP camera detection requires a compatible and accessible stream.
- Cloud deployment may require modifications for webcam access and audio alerts.
- The system is a demonstration project and should not be treated as a substitute for dedicated driver-safety equipment.

## 🚀 Future Enhancements

- Improve detection robustness under different lighting conditions.
- Add configurable EAR thresholds and frame limits.
- Add more flexible alert options.
- Display drowsiness statistics and event counts.
- Improve the user interface and result visualization.
- Evaluate performance using precision, recall, and other appropriate metrics.

## 🏁 Conclusion

The Drowsiness Detection System demonstrates how computer vision can be used to monitor eye activity and identify potential signs of drowsiness. By combining dlib facial landmarks, Eye Aspect Ratio calculations, OpenCV image processing, and a Streamlit interface, the project provides a practical example of fatigue-monitoring technology.

## 👨‍💻 Author

**Deep Menpara**

B.Tech in Information Technology

GitHub: [Deep7133](https://github.com/Deep7133)

## 📄 License

This project is intended for educational and demonstration purposes. Add an appropriate open-source license if you want others to reuse, modify, and distribute the project.

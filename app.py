import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance
import tempfile
# import winsound
COUNTER = 0
import os
import bz2
import urllib.request



# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Drowsiness Detection",
    layout="wide"
)

# ---------------- DLIB ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "shape_predictor_68_face_landmarks.dat"
)

MODEL_URL = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# Download and extract the model if it does not exist
if not os.path.exists(MODEL_PATH):

    st.info("Downloading facial landmark model... Please wait.")

    compressed_path = os.path.join(
        BASE_DIR,
        "shape_predictor_68_face_landmarks.dat.bz2"
    )

    urllib.request.urlretrieve(
        MODEL_URL,
        compressed_path
    )

    with bz2.open(compressed_path, "rb") as source:
        with open(MODEL_PATH, "wb") as target:
            target.write(source.read())

    os.remove(compressed_path)

@st.cache_resource
def load_dlib_models(model_path):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    return detector, predictor


detector, predictor = load_dlib_models(MODEL_PATH)

# ---------------- EAR FUNCTION ----------------

def calculate_EAR(eye):

    A = distance.euclidean(eye[1], eye[5])

    B = distance.euclidean(eye[2], eye[4])

    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

# ---------------- DETECTION FUNCTION ----------------

def detect_drowsiness(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector(gray)

    for face in faces:

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # FACE RECTANGLE
        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (255,0,255),
            2
        )

        # LANDMARKS
        shape = predictor(gray, face)

        shape = face_utils.shape_to_np(shape)

        # LEFT EYE
        leftEye = shape[42:48]

        # RIGHT EYE
        rightEye = shape[36:42]

        # DRAW EYE POINTS
        for (x,y) in leftEye:

            cv2.circle(
                frame,
                (x,y),
                2,
                (0,255,0),
                -1
            )

        for (x,y) in rightEye:

            cv2.circle(
                frame,
                (x,y),
                2,
                (0,255,0),
                -1
            )

        # EAR
        leftEAR = calculate_EAR(leftEye)

        rightEAR = calculate_EAR(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

       # LABEL

        global COUNTER

        if ear < 0.22:

            COUNTER += 1

        else:

            COUNTER = 0


        if COUNTER >= 20:

            label = "DROWSY ALERT!"

            color = (0,0,255)

            # PLAY ONLY ONCE
            if COUNTER == 20:
                st.warning("⚠️ Drowsiness detected! Please take a break.")

                # winsound.Beep(2500, 1000)

        else:

            label = "AWAKE"

            color = (0,255,0)

        # TEXT
        cv2.putText(
            frame,
            label,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    return frame

# ---------------- SIDEBAR ----------------
options = [
    "Home",
    "Image",
    "Video",
    "Camera",
    "URL"
]

if "menu" not in st.session_state:
    st.session_state.menu = "Home"

choice = st.sidebar.selectbox(
    "Select Activity",
    options,
    index=options.index(st.session_state.menu)
)

st.session_state.menu = choice

# ---------------- HOME ----------------

if choice == "Home":

    # TITLE
    st.title("😴 Drowsiness Detection System")

    st.markdown("""
       #### Real-Time Eye Monitoring using AI & Computer Vision
    """, unsafe_allow_html=True)

    # IMAGE
    st.image(
        "https://images.unsplash.com/photo-1516321318423-f06f85e504b3",
        width=500
    )

    # TECHNOLOGIES
    st.markdown("### ⚡ Technologies Used")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.success("Python")

    with t2:
        st.success("OpenCV")

    with t3:
        st.success("Dlib")

    with t4:
        st.success("Streamlit")
    
    if st.button("🚀 Start Detection"):

        st.session_state.menu = "Camera"

        st.rerun()
# ---------------- IMAGE ----------------

elif choice == "Image":

    file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if file:

        bytes_data = file.getvalue()

        np_array = np.frombuffer(
            bytes_data,
            np.uint8
        )

        img = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        result = detect_drowsiness(img)

        st.image(
            result,
            channels="BGR",
            width=900
        )

# ---------------- VIDEO ----------------

elif choice == "Video":

    file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if file:

        temp_file = tempfile.NamedTemporaryFile(
            delete=False
        )

        temp_file.write(file.read())

        cap = cv2.VideoCapture(
            temp_file.name
        )

        frame_window = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            result = detect_drowsiness(frame)

            frame_window.image(
                result,
                channels="BGR"
            )

        cap.release()

# ---------------- CAMERA ----------------
elif choice == "Camera":

    picture = st.camera_input("Take a picture")

    if picture is not None:

        bytes_data = picture.getvalue()

        np_array = np.frombuffer(bytes_data, np.uint8)

        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        result = detect_drowsiness(img)

        st.image(result, channels="BGR")     
# ---------------- URL ----------------

elif choice == "URL":

    url = st.text_input(
        "Enter WebCam URL"
    )

    if url:

        cap = cv2.VideoCapture(url)

        frame_window = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            result = detect_drowsiness(frame)

            frame_window.image(
                result,
                channels="BGR"
            )

        cap.release()

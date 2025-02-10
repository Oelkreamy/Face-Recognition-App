import streamlit as st
import cv2
import numpy as np
import face_recognition
import os

# Streamlit Page Config
st.set_page_config(page_title="Real-Time Face Recognition", layout="wide")

st.title("🔍 Real-Time Face Recognition App")
st.write("Upload known faces, then start the webcam to recognize them.")

# Load Known Faces
path = "team_pic"
if not os.path.exists(path):
    os.makedirs(path)

images = []
classNames = []

# Read images from folder
for cls in os.listdir(path):
    curImg = cv2.imread(os.path.join(path, cls))
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

# Function to Encode Faces
def find_encoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])  # Append encoding
    return encodeList

if images:
    encodeListKnown = find_encoding(images)
    st.success(f"✅ Loaded {len(images)} known faces")
else:
    st.warning("⚠️ No known faces found! Upload images first.")

# Upload new images
st.subheader("📤 Upload Face Images")
uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    st.success("✅ New images added! Refresh to load them.")

# Start Webcam
st.subheader("🎥 Live Face Recognition")
start_cam = st.button("Start Webcam")

# UI Containers for Displaying Recognized Name & ID
name_container = st.empty()
id_container = st.empty()
frame_window = st.image([])

TOLERANCE = 0.50  # Set face recognition threshold

def recognize(frame, tolerance):
    """Recognize face in the frame and return modified image, name, and ID"""
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    name = "Unknown"
    id_num = "N/A"

    for encodeFace, faceLoc in zip(encodeCurrFrame, facesCurrFrame):
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < tolerance:
            name = classNames[matchIndex].upper()
            id_num = f"ID-{matchIndex + 1}"  # Example ID format

        # Scale up face locations
        y1, x2, y2, x1 = [v * 4 for v in faceLoc]

        # Draw Rectangle & Label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return frame, name, id_num

if start_cam:
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("❌ Failed to capture frame from camera!")
            st.info("Please turn off other apps using the camera and restart the app.")
            st.stop()

        # Process frame
        image, name, id_num = recognize(frame, TOLERANCE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display results
        name_container.info(f"**Name:** {name}")
        id_container.success(f"**ID:** {id_num}")
        frame_window.image(image)

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
    if curImg is not None:
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

# Encode known faces
encodeListKnown = find_encoding(images) if images else []

if images:
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

if start_cam:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_window = st.image([])  # Streamlit image placeholder

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.error("❌ Failed to access webcam!")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize for faster processing
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurrFrame = face_recognition.face_locations(imgS)
        encodeCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

        for encodeFace, faceLoc in zip(encodeCurrFrame, facesCurrFrame):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis) if len(faceDis) > 0 else None

            name = "Unknown"
            if matchIndex is not None and faceDis[matchIndex] < 0.50:
                name = classNames[matchIndex].upper()

            # Scale up face locations
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]

            # Draw Rectangle & Label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Show smooth video
        frame_window.image(img, channels="BGR", use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

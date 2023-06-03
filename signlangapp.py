import cv2
import streamlit as st
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Set the title for the Streamlit app
st.title("Sign Language Recognition")

# Load the trained classifier model and labels
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Create an instance of HandDetector
detector = HandDetector(maxHands=1)

# Create a placeholder for displaying the sign letter
sign_letter_placeholder = st.empty()

# Add a "Stop" button and store its state in a variable
stop_button_pressed = st.button("Stop")

# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()

    if not ret:
        st.write("The video capture has ended.")
        break

    # Find hands using HandDetector
    hands, frame = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((300, 300, 3), np.uint8) * 255
        imgCrop = frame[y - 20:y + h + 20, x - 20:x + w + 20]

        # Resize the cropped image
        imgResize = cv2.resize(imgCrop, (300, 300))

        # Get prediction and index from the classifier
        prediction, index = classifier.getPrediction(imgResize, draw=False)
        sign_letter = classifier.getLabel(index)

        # Display the sign letter
        sign_letter_placeholder.text(sign_letter)

    # Display the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, channels="RGB")

    # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
        break

cap.release()
cv2.destroyAllWindows()

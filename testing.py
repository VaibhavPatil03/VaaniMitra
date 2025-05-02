import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Load trained model
model = load_model('Model/keras_model.h5')
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]

# Initialize detectors
detector = HandDetector(maxHands=1)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize and flip
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Use CVZone ---
    hands_cvzone, img_cv = detector.findHands(img.copy(), draw=True)

    # --- Use MediaPipe ---
    results = hands.process(imgRGB)

    features = []

    # Get MediaPipe landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])  # 3D points
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Get CVZone features (bounding box center, finger states, etc.)
    if hands_cvzone:
        hand = hands_cvzone[0]
        bbox = hand['bbox']
        center = hand['center']
        fingers = detector.fingersUp(hand)
        features.extend(center)
        features.extend(fingers)

    # Crop the hand region if available from CVZone
    if hands_cvzone:
        hand = hands_cvzone[0]
        x, y, w, h = hand['bbox']

        # Validate bbox coordinates to be within image bounds
        height, width = img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = max(0, w)
        h = max(0, h)
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y

        hand_img = img[y:y+h, x:x+w]

        # Check if hand_img is not empty before resizing
        if hand_img.size == 0:
            continue

        # Resize to model input size
        hand_img = cv2.resize(hand_img, (224, 224))
        hand_img = hand_img / 255.0  # Normalize
        hand_img = np.expand_dims(hand_img, axis=0)

        # Predict with model
        prediction = model.predict(hand_img, verbose=0)
        class_id = np.argmax(prediction)
        sign = labels[class_id]
        confidence = prediction[0][class_id]

        # Display
        cv2.putText(img, f"{sign} ({confidence:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)


    cv2.imshow("Combined Sign Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#(it is working and need few improement like w can use roi for more easy annd accurate results)
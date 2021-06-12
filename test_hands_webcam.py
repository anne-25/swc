import cv2 as cv
import mediapipe as mp
import sys

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cam_id = int(sys.argv[1])
cap = cv.VideoCapture(cam_id)
with mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         for p in range(21):
    #             print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv.imshow('MediaPipe Hands', image)
    if cv.waitKey(10) & 0xFF == 27:
      break
cap.release()

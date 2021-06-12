import cv2 as cv
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [720, 1280]

camera_id_to_loc = {0:4, 1:10}


def record_video_2_cams():
    #input video stream
    cap0 = cv.VideoCapture(camera_id_to_loc[0])
    cap1 = cv.VideoCapture(camera_id_to_loc[1])
    caps = [cap0, cap1]
    #set camera resolution
    for cap in caps:
        cap.set(3, 1280)
        cap.set(4, 720)

    #output video stream
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    rec0 = cv.VideoWriter('test/test_cam0.mp4',fourcc, 30.0, (720,720))
    rec1 = cv.VideoWriter('test/test_cam1.mp4',fourcc, 30.0, (720,720))

    recs = [rec0, rec1]

    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =1, min_tracking_confidence=0.5)

    kpts_cam0 = []
    kpts_cam1 = []

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        #crop to 720x720
        frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
        frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = hands0.process(frame0)
        results1 = hands1.process(frame1)

        #prepare list of hand keypoints of this frame
        #frame0 kpts
        frame0_keypoints = []
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                for p in range(21):
                    #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(round(frame0.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(round(frame0.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame0_keypoints.append(kpts)

        else:#no hand keypoints found
            frame0_keypoints = [[-1, -1]]*21 # set everything to dummy value

        kpts_cam0.append(frame0_keypoints)

        #frame1 kpts
        frame1_keypoints = []
        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                for p in range(21):
                    #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(round(frame1.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(round(frame1.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame1_keypoints.append(kpts)

        else:
            frame1_keypoints = [[-1, -1]]*21

        kpts_cam1.append(frame1_keypoints)


        # Draw the hand annotations on the image.
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        if results0.multi_hand_landmarks:
          for hand_landmarks in results0.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        rec0.write(frame0)

        if results1.multi_hand_landmarks:
          for hand_landmarks in results1.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        rec1.write(frame1)
        cv.imshow('cam1', frame1)
        cv.imshow('cam0', frame0)

        if cv.waitKey(1) & 0xFF == 27:
          break

    cv.destroyAllWindows()
    for cap, rec in zip(caps, recs):
        cap.release()
        rec.release()

    return np.array(kpts_cam0), np.array(kpts_cam1)

def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
        fout.write('\n')

    fout.close()

if __name__ == '__main__':

    kpts_cam0, kpts_cam1 = record_video_2_cams()

    write_keypoints_to_disk('test/kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('test/kpts_cam1.dat', kpts_cam1)

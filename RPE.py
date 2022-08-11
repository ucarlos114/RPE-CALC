##### INSTALL DEPENDENCIES #####
import mediapipe as mp
import cv2
import os
import numpy as np
from calculate_angle import *

##### SOLUTIONS (TRAINED ML ALGORITHMS?) #####
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


##### MAKE DETECTIONS FROM lIVE WEBCAM FEED #####
def live_detect():
    cap = cv2.VideoCapture(0)

    # Inititate model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()     # read input, returns two things, only need frame though

            # Recolor feed to RGB for processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Right hand
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_pose.HAND_CONNECTIONS)

            # Left hand
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_pose.HAND_CONNECTIONS)

            # Pose detection
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # display each processed frame
            cv2.imshow('Processed Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):      # press q to quit
                break

    cap.release()       # end live feed capture
    cv2.destroyAllWindows()     # close window


###### RUN #####

# live_detect()
# squat_detect('squat.mp4')
# video_detection('bench.mp4')
# video_detection('alex.MOV')

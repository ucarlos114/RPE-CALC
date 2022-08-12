##### INSTALL DEPENDENCIES #####
import mediapipe as mp
import cv2
import os
from calculate_angle import *

##### SOLUTIONS (TRAINED ML ALGORITHMS?) #####
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

##### DETECT FROM VIDEOS #####


def deadlift_detect(file):
    # setup video
    cap = cv2.VideoCapture(os.path.join(file))

    # Properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scale_percent = 40
    scale_width = int(width * (scale_percent / 100))
    scale_height = int(height * (scale_percent / 100))

    # Video Writer
    pathname = "output_videos/deadlift_output.avi"
    video_writer = cv2.VideoWriter(os.path.join(pathname), cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), fps, (scale_width, scale_height))

    # get funny input
    
    # Inititate holistic model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()     # read input, returns two things, only need frame though

            # resize image
            frame = cv2.resize(frame, (scale_width, scale_height),
                               interpolation=cv2.INTER_LINEAR)

            # Recolor feed to RGB for processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detections
            results = pose.process(image)
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                
                left_angle = np.around(calculate_angle(
                    left_shoulder, left_hip, left_knee), decimals=1, out=None)

                right_angle = np.around(calculate_angle(
                    right_shoulder, right_hip, right_knee), decimals=1, out=None)

                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                # Visualize
                cv2.putText(image, str(left_angle), tuple(np.multiply(left_hip, [int(1.1 * scale_width), scale_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_angle), tuple(np.multiply(right_hip, [int(0.8 * scale_width), scale_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                '''headline = str(weight) + " solid ass pounds"
                cv2.putText(image, headline, tuple(np.multiply(nose, [int(0.2*scale_width), int(0.8*scale_height)]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, headline, tuple(np.multiply(nose, [int(0.2*scale_width), int(0.8*scale_height)]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)'''

            except:
                pass

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(200, 56, 83), thickness=2, circle_radius=2))

            # display each processed frame
            cv2.imshow('Processed Feed', image)

            # Write out frame
            video_writer.write(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):      # press q to quit
                break

    cap.release()       # end live feed capture
    cv2.destroyAllWindows()     # close window
    # Release video writer
    video_writer.release()

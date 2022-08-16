##### INSTALL DEPENDENCIES #####
from audioop import add
from turtle import update
import mediapipe as mp
import cv2
import os
from calculate_angle import *
from update_db import *
#from convert import *

##### SOLUTIONS (TRAINED ML ALGORITHMS?) #####
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

##### DETECT FROM VIDEOS #####

def get_fps(file):
    # setup video
    cap = cv2.VideoCapture(os.path.join(file))

    # Properties
    fps = round(cap.get(cv2.CAP_PROP_FPS), 3)
    print(str(fps))

# -------------------------------------------------------------------

def test_deadlift_detect(file, rpe):

    # setup video
    cap = cv2.VideoCapture(os.path.join(file))

    # Properties
    fps = round(cap.get(cv2.CAP_PROP_FPS), 3)
    scale_width = 576
    scale_height = 768

    # Create Video Writer
    output = file.split('/', 1)[1].split('.', 1)[0]
    video_writer = cv2.VideoWriter(os.path.join("output_videos/" + output + ".avi"), cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), fps, (scale_width, scale_height))

    # set variables
    numframes = 0           # number of frames elapsed (later to be frames elasped since start of lift)
    reps = 0                # number of reps completed
    max_ha = 0              # max hip angle
    min_ha = 180            # min hip angle
    max_ka = 0              # max knee angle
    min_ka = 180            # min knee angle
    max_rom = 0             # max distance between hands and ankles
    min_rom = scale_height  # min distance between hands and ankles
    prev_rom = 0            # for keeping track of upward motion
    complete = False        # whether the bar is currently at the top (rep complete)
    time = 0

    # Inititate holistic model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():           # loop through all frames
            ret, frame = cap.read()     # read input, returns two things, only need frame though

            # resize frame
            try:
                frame = cv2.resize(
                    frame, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                break

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
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
                right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]

                # data to be collected
                ha = round((calculate_angle(left_shoulder, left_hip, left_knee) + 
                    calculate_angle(right_shoulder, right_hip, right_knee)) / 2, 2)

                ka = round((calculate_angle(left_hip, left_knee, left_ankle) +
                    calculate_angle(right_hip, right_knee, right_ankle)) / 2, 2)

                rom = ( (left_ankle[1] - left_hand[1]) +  (right_ankle[1] - right_hand[1]) ) / 2

                # update max/min values
                if ha > max_ha:
                    max_ha = ha
                elif ha < min_ha:
                    min_ha = ha

                if ka > max_ka:
                    max_ka = ka
                elif ka < min_ka:
                    min_ka = ka

                if rom > max_rom:
                    max_rom = rom
                elif rom < min_rom:
                    min_rom = rom
                
                # check if rep has been completed
                if (ka > 170 and ha > 165 and not complete):
                    reps += 1
                    complete = True
                    upward = False
                elif (ha < 100):
                    complete = False

                # check if rep is in progress
                if (rom > prev_rom and not complete):
                    cv2.putText(image, 'LIFTING', [20, 25], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                    numframes += 1      # frame counter
                prev_rom = rom
                
                #timer = "Time Elapsed: " + '{:.2f}'.format(time)
                
                counter = "Reps completed: " + str(reps)
                cv2.putText(image, counter, [20, 55], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(image, str(ha), tuple(np.multiply(left_hip, [int(1.1 * scale_width), scale_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(ha), tuple(np.multiply(right_hip, [int(0.8 * scale_width), scale_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(ka), tuple(np.multiply(left_knee, [int(1.1 * scale_width), scale_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(ka), tuple(np.multiply(right_knee, [int(0.8 * scale_width), scale_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(image, str(round(rom, 2)), tuple(np.multiply(left_hand, [int(1.1 * scale_width), scale_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(rom, 2)), tuple(np.multiply(right_hand, [int(0.8 * scale_width), scale_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            except:
                pass

            '''mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(200, 56, 83), thickness=2, circle_radius=2))'''

            # display each processed frame
            cv2.imshow('RPE Calculator', image)

            # Write out frame
            video_writer.write(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):      # press q to quit
                break

    # convert output vid to mp4
    #video_to_mp4('output_videos/' + output + '.avi', output='output_videos/' + output + '.mp4')
    
    # final calculations
    time = numframes / fps
    hip_av = round((max_ha - min_ha) / time, 1)   # average hip angular velocity
    knee_av = round((max_ka - min_ka) / time, 1)  # average knee angular velocity
    bar_speed = round((max_rom - min_rom) / time, 1) # average upward velocity of the barbell (NEED TO FIND BETTER METHOD OF DETERMINING)

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()

    # print results (later write to a csv)
    print('Time elapsed was: ' + str(round(time, 2)))
    print('Average hip angular velocity was: ' + str(hip_av) + ' degrees per second')
    print('Average knee angular velocity was: ' + str(knee_av) + ' degrees per second')
    print('Average bar speed was: ' + str(bar_speed))
    print('RPE ' + str(rpe))
    add_to_data(output, time, reps, hip_av, knee_av, bar_speed, rpe)

    return(output, time, reps, hip_av, knee_av, bar_speed, rpe)
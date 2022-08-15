import mediapipe as mp
import cv2
import os

def video_to_mp4(input, output, fps: int = 0, frame_size: tuple = (), fourcc: str = "MPV4"):
    vidcap = cv2.VideoCapture(input)
    if not fps:
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    success, arr = vidcap.read()
    if not frame_size:
        height, width, _ = arr.shape
        frame_size = width, height
    writer = cv2.VideoWriter(
        output,
        apiPreference=0,
        fourcc=cv2.VideoWriter_fourcc(*fourcc),
        fps=fps,
        frameSize=frame_size
    )
    while True:
        if not success:
            break
        writer.write(arr)
        success, arr = vidcap.read()
    writer.release()
    vidcap.release()

# video_to_mp4('input_videos/jacob_sumo.MOV', output='output_videos/jacob_sumo.mp4')
video_to_mp4('input_videos/435_pause.MOV', output='input_videos/435_pause.mp4')
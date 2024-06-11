import cv2
from time import time
import mediapipe as mp
import numpy as np
import json

cap = cv2.VideoCapture(0) # Capture video from camera

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

cap.set(3, width)
cap.set(4, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))


# initialize media pipe model to detect hand movement 
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
connection_drawing_spec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

# initializtion for timesteps list for the video
first_time = None
video_timestamps = []

# initialization for landmark list. Non-standard height and width
landmarks_v_time = [] 



while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,1)

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # process to RGB form
        results = hands.process(imgRGB) # result of hand gesture decoding

        # record and save timestamps
        if not first_time:
            first_time = time()
            video_timestamps.append(0)
        else:
            video_timestamps.append(time() - first_time)

        
        # save the gesture points
        if results.multi_hand_landmarks:
            a = []
            for handLms in results.multi_hand_landmarks: 
                b = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c =frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    b.append([cx, cy])
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    mpDraw.draw_landmarks(
                        frame, 
                        handLms, 
                        mpHands.HAND_CONNECTIONS,
                        mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                        connection_drawing_spec
                    )
                a.append(b)
            landmarks_v_time.append(a)

        

        cv2.imshow('frame',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    else:
        break

# Release everything if job is finished
out.release()
cap.release()
cv2.destroyAllWindows()


landmarks_data = {
    'timestamps': video_timestamps,
    'landmarks': landmarks_v_time
}

with open('landmark_data.json', 'w') as json_file:
    json.dump(landmarks_data, json_file)

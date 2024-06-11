import numpy as np
import cv2
from time import time
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0) # Capture video from camera

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

cap.set(3, width)
cap.set(4, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))


# 



# initializtion for timesteps for the video
first_time = None
video_timestamps = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        # record and save timestamps
        if not first_time:
            first_time = time()
            video_timestamps.append(0)
        else:
            video_timestamps.append(time() - first_time)

        cv2.imshow('frame',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    else:
        break

# Release everything if job is finished
out.release()
cap.release()
cv2.destroyAllWindows()

timestamps_array = np.array(video_timestamps)
np.save('video_timestamps.npy', timestamps_array)

import cv2
import mediapipe as mp
import time
import numpy as np


frameWidth = 1280
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

connection_drawing_spec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

while True:
    success, img = cap.read()
    img= cv2.flip(img,1)
    img_w = np.empty(img.shape)
    img_w.fill(255)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)



    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: 
        #handLMs are 21 points. so we need conection too-->mpHands.HAND_CONNECTIONS
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                #lm = x,y cordinate of each landmark in float numbers. lm.x, lm.y methods
                #So, need to covert in integer
                h, w, c =img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                # if id == 4: #(To draw 4th point)
                cv2.circle(img_w, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(
                img_w, 
                handLms, 
                mpHands.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec
            ) #drawing points and lines(=handconections)

    #Write frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img_w, "FPS= " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 0), 1)
    
    cv2.imshow('image', img_w)
    if cv2.waitKey(1)==27:
        break
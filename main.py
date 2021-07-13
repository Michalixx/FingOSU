import cv2
import time
import mediapipe as mp
import HandTrackingModule as htm
import math
import random

pTime = 0
cTime = 0

cap = cv2.VideoCapture(1)
detector = htm.HandDetector(mode = False, maxHands = 1, detectionCon = 0.7, trackCon = 0.5)

circle = 0, 0
points = 0


def collision(p1, p2, d):
    distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return distance <= d


# [4, 225, 305] finger
def game(img, finger):
    global points, circle
    x, y = finger[1], finger[2]
    cv2.circle(img, circle, 20, (255, 0, 0), 3)  # Settings here
    if collision((x, y), circle, 25):
        points += 1
        circle = (random.randrange(20, img.shape[1]-50), random.randrange(20, img.shape[0]-50))
        print(img.shape)
        game(img, finger)


while True:
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)

        img = detector.findHands(img, draw=False)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) == 0:
            finger = (0, 0, 0)
        else:
            finger = lmList[8]
            cv2.circle(img, (lmList[8][1], lmList[8][2]), 5, (0, 0, 255), 3)
        game(img, finger)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(points), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
    cv2.waitKey(1)
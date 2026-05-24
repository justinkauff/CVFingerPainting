"""
Vision-Based Finger Painting System
Author: Justin Kauff

Interactive real-time computer vision application that transforms webcam
input into a virtual painting canvas using fingertip tracking and object-aware
brush control.

Features:
- HSV-based fingertip segmentation
- Morphological filtering and contour extraction
- Centroid tracking using image moments
- Dynamic brush sizing based on contour area
- YOLOv8 object recognition for real-time color selection
- Live webcam/canvas overlay rendering
- Canvas export and interactive controls

Technologies:
Python, OpenCV, NumPy, YOLOv8 (Ultralytics)

Copyright (c) 2026 Justin Kauff
Released for portfolio and educational purposes.
"""

# adding in YoloV8 for object detection - for color selection
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# MAKE SURE INSTRUCTION.PNG EXISTS IN SAME FOLDER AS SCRIPT
script_dir = os.path.dirname(os.path.abspath(__file__))
instructions = cv2.imread(os.path.join(script_dir, 'instruction.png'))
instruction_open = False

# import v8 model
model = YOLO("yolov8n.pt") # nano = fastest, good for real time

# setup the window
cv2.namedWindow('preview')
vc= cv2.VideoCapture(0)


# check if webcam is readable/connected
if not vc.isOpened():
    print("Could not open webcam")
    exit()

# get the first frame to get h x w, used for np array
rval, frame = vc.read()
h, w = frame.shape[:2]
print(f"Webcam resolution: {w} * {h}")

# create the blank canvas (same dimensions as webcam, black background)
canvas = np.zeros((h,w,3), dtype = np.uint8)

# drawing state
prev_point = None
drawing = True # toggelable with 'd'
brush_color = (0,0,255) # red, BGR
brush_size = 8

# HSV thresholding, detect blue
BLUE_LOWER = np.array([100,120,70])
BLUE_UPPER = np.array([130,255,255])

# frame counter for object detection
frame_count = 0

color_change_msg = ""
object_detection_msg =""
color_change_timer = 0

# MAP FOR COLORS
OBJECT_COLOR_MAP = {
    "cell phone":  (255, 0, 0),     # blue
    "stop sign":   (0, 0, 255),     # red (fitting)
    "cup":         (0, 255, 0),     # green
    "bottle":      (0, 255, 255),   # yellow
    "remote":      (255, 0, 255),   # magenta
    "laptop":      (0, 165, 255),   # orange
}
COLOR_NAME_MAP = {
    "cell phone":  "Blue",
    "stop sign":   "Red",
    "cup":         "Green",
    "bottle":      "Yellow",
    "remote":      "Magenta",
    "laptop":      "Orange",
}

while True:
    rval, frame = vc.read()

    if not rval:
        print("Frame not received. Exiting...")
        break
    # mirror the image
    frame = cv2.flip(frame,1)

    # detect blue finger
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    # clean up the mask
    kernel = np.ones((5,5),np.uint8)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)   # kill noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # fill holes
    mask = cv2.erode(mask, None, iterations = 1)
    mask = cv2.dilate(mask, None, iterations = 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finger_pos = None

    if contours:
        # take largest contour
        largest = max(contours, key=cv2.contourArea)

        # Only track if the blob is big enough
        # 5/13 adding code to change brush size based on distance
        # idea: smaller blob size means finger further away, = smaller brush
        area = cv2.contourArea(largest)
        if area > 50:
            M = cv2.moments(largest) # dictonary of weighted pixel sums
            # divide by zero handler
            if M["m00"] != 0:
                
                '''
                m00 - zeroth moment, total area
                m10 - sum of all pixel x coords, weighted by intensity
                m01 - same thing but for y coords
                '''

                cx = int(M["m10"] / M["m00"]) # average x
                cy = int(M["m01"] / M["m00"]) # average y
                finger_pos = (cx,cy)

                # visual feedback: circle on webcam feed
                cv2.circle(frame, finger_pos, 10, (0, 255, 0), -1)

                # map area to brush size
                brush_size = int(np.interp(area, [200, 1000], [2, 30]))

    if finger_pos and drawing:
        if prev_point:
            cv2.line(canvas, prev_point, finger_pos, brush_color, brush_size)
        prev_point = finger_pos
    else:
        prev_point = None

    # object detection, run every tenth frame because its very slow

    if frame_count % 10 == 0:
        results = model(frame, verbose = False)
        result = results[0]
        for det in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            label = model.names[int(cls)]

            # draw bounding box for everything detected - looks weird bc runs every 10th frame, disregarding this
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            # cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if label in OBJECT_COLOR_MAP and conf > 0.65:
                brush_color = OBJECT_COLOR_MAP[label]
                color_change_msg = f"Color changed to {COLOR_NAME_MAP[label]}!"
                object_detection_msg = f"Object detected = {label}"
                color_change_timer = 60  # frames (~2 sec at 30fps)
                break

    # overlay canvas on webcam feed
    # only show canvas pixels where something has been drawn
    canvas_mask = canvas.astype(bool).any(axis=2)
    combined = frame.copy()
    combined[canvas_mask] = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)[canvas_mask]

    # UI text/instructions
    status = "DRAWING" if drawing else "PAUSED"
    cv2.putText(combined, f"[{status}] D=toggle | C=clear | I=colors | S=save | ESC=quit",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(combined, f"Brush: {brush_size}px", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, brush_color, 2)
    
    # show what has been detected and what color everything has been changed to
    if color_change_timer > 0:
        cv2.putText(combined, object_detection_msg, (w//2 - 150, h//2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(combined, color_change_msg, (w//2 - 150, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, brush_color, 3)
        color_change_timer -= 1

    cv2.imshow('preview', frame)
    cv2.imshow('Finger Painting', combined)
    cv2.imshow('Blue Mask', mask)
    cv2.imshow('Canvas Only', canvas)

    
    # all the keyboard inputs
    key = cv2.waitKey(20)
    if key == 27:
        break
    elif key == ord('c'):
        canvas = np.zeros((h,w,3), dtype = np.uint8)
    elif key == ord('d'):
        drawing = not drawing
    elif key == ord('i'):
        if instructions is None:
            print("Could not load instruction.png")
            continue
        if not instruction_open:
            cv2.imshow('Color Change Instructions', instructions)
            instruction_open = True
        else:
            cv2.destroyWindow('Color Change Instructions')
            instruction_open = False
    elif key == ord('s'):
        # getTickCount so file name is unique
        filename = f"canvas_{int(cv2.getTickCount())}.png"
        cv2.imwrite(os.path.join(script_dir, filename), canvas)
        print(f"Canvas saved as {filename}")
    frame_count += 1

vc.release()
cv2.destroyAllWindows()
# Vision-Based Finger Painting System

Interactive real-time computer vision application that transforms webcam input into a virtual finger-painting canvas using fingertip tracking, HSV segmentation, and YOLOv8 object recognition.

## Overview

This project uses OpenCV and YOLOv8 to create a webcam-based drawing interface where a tracked fingertip acts as a virtual paintbrush. A colored fingertip is isolated through HSV thresholding and morphological filtering, then tracked frame-by-frame using contour analysis and image moments. The resulting centroid coordinates are used to generate smooth real-time drawing strokes on a digital canvas.

To extend interactivity, YOLOv8 object recognition dynamically changes the brush color when specific household objects are detected in the camera frame. Brush size also scales dynamically based on contour area, approximating finger depth relative to the camera.

## Features

* Real-time webcam finger tracking
* HSV color segmentation
* Morphological mask cleanup
* Contour detection and centroid extraction
* Dynamic brush sizing based on contour area
* YOLOv8 object detection for brush color selection
* Live drawing overlay and standalone canvas rendering
* Canvas export functionality
* Interactive keyboard controls

## Technologies

* Python
* OpenCV
* NumPy
* YOLOv8 (Ultralytics)

## Object-Based Color Mapping

| Object Detected | Brush Color |
| --------------- | ----------- |
| Cell Phone      | Blue        |
| Stop Sign       | Red         |
| Cup             | Green       |
| Bottle          | Yellow      |
| Remote          | Magenta     |
| Laptop          | Orange      |

## Controls

| Key | Action                    |
| --- | ------------------------- |
| D   | Toggle drawing            |
| C   | Clear canvas              |
| I   | Toggle instruction window |
| S   | Save canvas image         |
| ESC | Quit application          |

## How It Works

1. Webcam frames are captured in real time using OpenCV.
2. Frames are converted from BGR to HSV color space.
3. HSV thresholding isolates the colored fingertip.
4. Morphological operations remove noise and fill gaps in the mask.
5. Contours are extracted and the largest contour is selected as the fingertip.
6. Image moments compute the centroid location of the contour.
7. Consecutive centroid positions are connected with lines to generate drawing strokes.
8. YOLOv8 periodically runs object detection to modify brush color dynamically.
9. Contour area is mapped to brush thickness for depth-aware drawing behavior.

## Future Improvements

* Gesture recognition for brush/tool switching
* Shape snapping and stroke smoothing
* Multi-finger tracking
* Custom color palette selection
* GPU acceleration for faster inference

## Author

Justin Kauff

Copyright (c) 2026 Justin Kauff

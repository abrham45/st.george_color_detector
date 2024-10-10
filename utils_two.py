# util.py
import numpy as np
import cv2

def get_limits(color_name):
    if color_name == "yellow":
        # Slightly lowered thresholds for yellow
        lowerLimit = np.array([20, 180, 180], dtype=np.uint8)  # Lowered saturation and value
        upperLimit = np.array([30, 255, 255], dtype=np.uint8)
    elif color_name == "red":
        # Slightly lowered thresholds for red
        lowerLimit1 = np.array([0, 180, 180], dtype=np.uint8)
        upperLimit1 = np.array([10, 255, 255], dtype=np.uint8)
        lowerLimit2 = np.array([170, 180, 180], dtype=np.uint8)
        upperLimit2 = np.array([180, 255, 255], dtype=np.uint8)
        return (lowerLimit1, upperLimit1), (lowerLimit2, upperLimit2)
    else:
        raise ValueError("Color not supported")

    return lowerLimit, upperLimit

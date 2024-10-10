import numpy as np
import cv2

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Set very tight limits for pure red color detection
    if hue >= 165:  # Upper limit for red hue
        lowerLimit = np.array([160, 150, 150], dtype=np.uint8)  # Lower hue
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)  # Upper hue
    elif hue <= 15:  # Lower limit for red hue
        lowerLimit = np.array([0, 150, 150], dtype=np.uint8)  # Lower hue
        upperLimit = np.array([10, 255, 255], dtype=np.uint8)  # Upper hue
    else:
        lowerLimit = np.array([0, 150, 150], dtype=np.uint8)  # Lower limit for red
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)  # Upper limit for red

    return lowerLimit, upperLimit
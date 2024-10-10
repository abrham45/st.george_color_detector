import cv2
import numpy as np
from PIL import Image
from util import get_limits

# Load the pre-trained MobileNet SSD model for person detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

yellow = [0, 0, 255]  # Yellow in BGR colorspace
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")

overlay_image = cv2.imread('overlay.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Load the success image (to display on success)
success_image = cv2.imread('success.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Check if the images were loaded successfully
if overlay_image is None or success_image is None:
    print("Error: Could not load overlay or success image.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Prepare the frame for person detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Initialize person-detection variables
    person_detected = False
    person_box = None

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            # Class label for 'person' in the MobileNet SSD model is 15
            if idx == 15:
                person_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                person_box = (startX, startY, endX, endY)

                # Draw a rectangle around the detected person
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                break

    if person_detected and person_box is not None:
        # Extract the region of interest (ROI) where the person is located
        roi = frame[person_box[1]:person_box[3], person_box[0]:person_box[2]]

        # Convert the ROI to HSV color space for color detection
        hsvImage = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Get the color limits for yellow
        lowerLimit, upperLimit = get_limits(color=yellow)
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        # Detect yellow in the ROI (mask non-black areas)
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()

        # Resize overlay image to match frame size
        overlay_resized = cv2.resize(overlay_image, (frame.shape[1], frame.shape[0]))

        # Check if the overlay image has an alpha channel
        if overlay_resized.shape[2] == 4:  # Image with alpha channel
            b, g, r, a = cv2.split(overlay_resized)
            alpha_mask = a.astype(float) / 255.0

            for c in range(0, 3):  # For each color channel
                frame[:, :, c] = (alpha_mask * overlay_resized[:, :, c] +
                                  (1 - alpha_mask) * frame[:, :, c])

        if bbox is not None:
            # Resize the success image to a suitable size (e.g., 200x100 pixels)
            success_image_resized = cv2.resize(success_image, (frame.shape[1], 200))

            # Calculate position to center the success image at the bottom center
            x_offset = (frame.shape[1] - success_image_resized.shape[1]) // 2
            y_offset = frame.shape[0] - success_image_resized.shape[0] - 10  # 10 pixels from bottom

            # Check if the success image has an alpha channel
            if success_image_resized.shape[2] == 4:
                b, g, r, a = cv2.split(success_image_resized)
                alpha_mask = a.astype(float) / 255.0

                # Overlay the success image
                for c in range(0, 3):  # For each color channel
                    frame[y_offset:y_offset + success_image_resized.shape[0],
                          x_offset:x_offset + success_image_resized.shape[1], c] = (
                        alpha_mask * success_image_resized[:, :, c] +
                        (1 - alpha_mask) * frame[y_offset:y_offset + success_image_resized.shape[0],
                                                x_offset:x_offset + success_image_resized.shape[1], c])

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

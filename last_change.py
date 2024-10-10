import cv2
import numpy as np
from PIL import Image
from utils import get_limits
import time  # For timestamps

# Load the pre-trained MobileNet SSD model for person detection
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/mobilenet_iter_73000.caffemodel')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")

overlay_image = cv2.imread('assets/overlay.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Load the success images (to display based on conditions)
success_images = {
    'both': cv2.imread('assets/100.png', cv2.IMREAD_UNCHANGED),
    'either': cv2.imread('assets/50.png', cv2.IMREAD_UNCHANGED),
    'none': cv2.imread('assets/0.png', cv2.IMREAD_UNCHANGED)
}

# Check if the images were loaded successfully
if overlay_image is None or any(img is None for img in success_images.values()):
    print("Error: Could not load overlay or success images.")
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
                break

    # Initialize color-detection variables
    red_percentage = 0
    yellow_percentage = 0

    if person_detected and person_box is not None:
        startX, startY, endX, endY = person_box

        # Ensure the coordinates are within the frame boundaries
        if startX >= 0 and startY >= 0 and endX <= w and endY <= h:
            # Extract the region of interest (ROI) where the person is located
            roi = frame[startY:endY, startX:endX]

            # Convert the ROI to HSV color space for color detection
            hsvImage = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Get the color limits for red and yellow
            yellow_lower, yellow_upper = get_limits('yellow')
            (red_lower1, red_upper1), (red_lower2, red_upper2) = get_limits('red')

            # Detect yellow in the ROI
            yellow_mask = cv2.inRange(hsvImage, yellow_lower, yellow_upper)
            yellow_count = cv2.countNonZero(yellow_mask)

            # Detect red in the ROI
            red_mask1 = cv2.inRange(hsvImage, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsvImage, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_count = cv2.countNonZero(red_mask)

            # Total pixels in ROI
            total_pixels = roi.shape[0] * roi.shape[1]

            if total_pixels > 0:
                # Calculate the percentage of red and yellow
                red_percentage = (red_count / total_pixels) * 100
                yellow_percentage = (yellow_count / total_pixels) * 100

    # Display the percentage of detected colors on the frame
    text = f"Red: {red_percentage:.2f}% | Yellow: {yellow_percentage:.2f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Determine which success image to use
    if red_percentage > 0 and yellow_percentage > 0:
        success_image = success_images['both']
    elif red_percentage > 0 or yellow_percentage > 0:
        success_image = success_images['either']
    else:
        success_image = success_images['none']

    # Overlay the success image
    success_image_resized = cv2.resize(success_image, (200, 100))
    x_offset = (frame.shape[1] - success_image_resized.shape[1]) // 2
    y_offset = frame.shape[0] - success_image_resized.shape[0] - 10

    # Check if the success image has an alpha channel
    if success_image_resized.shape[2] == 4:
        b, g, r, a = cv2.split(success_image_resized)
        alpha_mask = a.astype(float) / 255.0

        for c in range(0, 3):  # For each color channel
            frame[y_offset:y_offset + success_image_resized.shape[0],
                  x_offset:x_offset + success_image_resized.shape[1], c] = (
                alpha_mask * success_image_resized[:, :, c] +
                (1 - alpha_mask) * frame[y_offset:y_offset + success_image_resized.shape[0],
                                        x_offset:x_offset + success_image_resized.shape[1], c])

    # Resize the overlay image to fit the frame dimensions
    overlay_height, overlay_width = overlay_image.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    # Resize overlay while maintaining aspect ratio
    if overlay_width > frame_width or overlay_height > frame_height:
        scale = min(frame_width / overlay_width, frame_height / overlay_height)
        new_size = (int(overlay_width * scale), int(overlay_height * scale))
        overlay_image = cv2.resize(overlay_image, new_size, interpolation=cv2.INTER_AREA)

    # Overlay the resized overlay image
    if overlay_image.shape[2] == 4:  # Check for alpha channel
        b, g, r, a = cv2.split(overlay_image)
        alpha_mask = a.astype(float) / 255.0

        # Calculate the position to place the overlay
        x_offset = (frame.shape[1] - overlay_image.shape[1]) // 2
        y_offset = (frame.shape[0] - overlay_image.shape[0]) // 2

        for c in range(0, 3):
            frame[y_offset:y_offset + overlay_image.shape[0],
                  x_offset:x_offset + overlay_image.shape[1], c] = (
                alpha_mask * overlay_image[:, :, c] +
                (1 - alpha_mask) * frame[y_offset:y_offset + overlay_image.shape[0],
                                          x_offset:x_offset + overlay_image.shape[1], c])

    cv2.imshow('frame', frame)

    # Capture photo on 'c' key press
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Get the current timestamp and format it
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'user_photo_{timestamp}.png'  # Use the timestamp in the filename
        cv2.imwrite(filename, frame)  # Save the current frame as an image
        print(f"Photo captured and saved as '{filename}'.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

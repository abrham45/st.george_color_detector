import cv2
import numpy as np
from PIL import Image
from utils import get_limits
import time

net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/mobilenet_iter_73000.caffemodel')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not open video capture.")

overlay_image = cv2.imread('assets/overlay.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

success_images = {
    '0': cv2.imread('assets/0.png', cv2.IMREAD_UNCHANGED),
    '25': cv2.imread('assets/25.png', cv2.IMREAD_UNCHANGED),
    '50': cv2.imread('assets/50.png', cv2.IMREAD_UNCHANGED),
    '75': cv2.imread('assets/75.png', cv2.IMREAD_UNCHANGED),
    '100': cv2.imread('assets/100.png', cv2.IMREAD_UNCHANGED)
}

if overlay_image is None or any(img is None for img in success_images.values()):
    print("Error: Could not load overlay or success images.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

def calculate_percentage(mask, total_pixels):
    return (cv2.countNonZero(mask) / total_pixels) * 100

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_detected = False
    person_box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:
                person_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                person_box = (startX, startY, endX, endY)
                break

    red_found = False
    yellow_found = False
    red_percentage = 0
    yellow_percentage = 0

    if person_detected and person_box is not None:
        startX, startY, endX, endY = person_box
        if startX >= 0 and startY >= 0 and endX <= w and endY <= h:
            roi = frame[startY:endY, startX:endX]
            hsvImage = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            yellow_lower, yellow_upper = get_limits('yellow')
            (red_lower1, red_upper1), (red_lower2, red_upper2) = get_limits('red')

            yellow_mask = cv2.inRange(hsvImage, yellow_lower, yellow_upper)
            red_mask1 = cv2.inRange(hsvImage, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsvImage, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            total_pixels = roi.shape[0] * roi.shape[1]
            yellow_percentage = calculate_percentage(yellow_mask, total_pixels)
            red_percentage = calculate_percentage(red_mask, total_pixels)

    # Calculate the overall detection percentage
    combined_percentage = max(red_percentage, yellow_percentage)

    # Determine which image to show based on percentage
    if combined_percentage <= 10:
        success_image = success_images['0']
    elif 11 <= combined_percentage <= 30:
        success_image = success_images['25']
    elif 31 <= combined_percentage <= 50:
        success_image = success_images['50']
    elif 51 <= combined_percentage <= 75:
        success_image = success_images['75']
    else:
        success_image = success_images['100']

    # Add percentage text to frame
    percentage_text = f'Red: {red_percentage:.2f}%, Yellow: {yellow_percentage:.2f}%, Combined: {combined_percentage:.2f}%'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, percentage_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Overlay success image
    success_image_resized = cv2.resize(success_image, (200, 100))
    x_offset = (frame.shape[1] - success_image_resized.shape[1]) // 2
    y_offset = frame.shape[0] - success_image_resized.shape[0] - 10

    if success_image_resized.shape[2] == 4:
        b, g, r, a = cv2.split(success_image_resized)
        alpha_mask = a.astype(float) / 255.0

        for c in range(0, 3):
            frame[y_offset:y_offset + success_image_resized.shape[0],
                  x_offset:x_offset + success_image_resized.shape[1], c] = (
                alpha_mask * success_image_resized[:, :, c] +
                (1 - alpha_mask) * frame[y_offset:y_offset + success_image_resized.shape[0],
                                        x_offset:x_offset + success_image_resized.shape[1], c])

    # Handle overlay
    overlay_height, overlay_width = overlay_image.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    if overlay_width > frame_width or overlay_height > frame_height:
        scale = min(frame_width / overlay_width, frame_height / overlay_height)
        new_size = (int(overlay_width * scale), int(overlay_height * scale))
        overlay_image = cv2.resize(overlay_image, new_size, interpolation=cv2.INTER_AREA)

    if overlay_image.shape[2] == 4:
        b, g, r, a = cv2.split(overlay_image)
        alpha_mask = a.astype(float) / 255.0

        x_offset = (frame.shape[1] - overlay_image.shape[1]) // 2
        y_offset = (frame.shape[0] - overlay_image.shape[0]) // 2

        for c in range(0, 3):
            frame[y_offset:y_offset + overlay_image.shape[0],
                  x_offset:x_offset + overlay_image.shape[1], c] = (
                alpha_mask * overlay_image[:, :, c] +
                (1 - alpha_mask) * frame[y_offset:y_offset + overlay_image.shape[0],
                                          x_offset:x_offset + overlay_image.shape[1], c])

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'user_photo_{timestamp}.png'
        cv2.imwrite(filename, frame)
        print(f"Photo captured and saved as '{filename}'.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

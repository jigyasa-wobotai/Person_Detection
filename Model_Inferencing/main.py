import os
import cv2

video_path = "main_court_floor_view2.mp4"
output_dir = "original_frames/ATP/main_court_floor_view2"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_number = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    filename = f"frame_{frame_number:06d}.jpg"
    filepath = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, frame)

    frame_number += 1

cap.release()

print(f"Extracted {frame_number} frames")

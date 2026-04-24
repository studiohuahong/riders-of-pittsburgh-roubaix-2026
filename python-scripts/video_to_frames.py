import cv2
import os
video_path = "video.mov"

output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_filename = os.path.join(
        output_folder, f"frame_{frame_count:06d}.jpg"
    )
    
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

print(f"Done! Extracted {frame_count} frames.")

cap.release()
import cv2
import numpy as np

# ========= CONFIG =========
video_path = "video.mov"
output_path = "slit_scan_strip_rtl.png"

column_ratio = 0.5   # 0.5 = center
slice_width = 70      # number of columns per frame
resize_scale = 1.0   # e.g. 0.3 if too large
frame_step = 1       # skip frames if needed
# ==========================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("Cannot open video")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frames: {frame_count}, Width: {width}, Height: {height}")

# choose column
column_x = int(width * column_ratio)

# define slice region
half_w = slice_width // 2
left = max(column_x - half_w, 0)
right = min(column_x + half_w + 1, width)

actual_slice_width = right - left
print(f"Using columns: {left} to {right} (width={actual_slice_width})")

# estimate output width
effective_frames = frame_count // frame_step
output_width = effective_frames * actual_slice_width

# allocate output image
output = np.zeros((height, output_width, 3), dtype=np.uint8)

frame_idx = 0
out_x = output_width  # start from RIGHT side

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_step != 0:
        frame_idx += 1
        continue

    # extract slice (NO averaging)
    slice_region = frame[:, left:right, :]  # (H, slice_width, 3)

    w = slice_region.shape[1]

    # move left and insert
    out_x -= w
    output[:, out_x:out_x + w, :] = slice_region

    frame_idx += 1

cap.release()

# trim (in case frame count estimate was off)
output = output[:, out_x:, :]

# optional resize
if resize_scale != 1.0:
    output = cv2.resize(output, None, fx=resize_scale, fy=resize_scale)

cv2.imwrite(output_path, output)

print(f"Saved to {output_path}")
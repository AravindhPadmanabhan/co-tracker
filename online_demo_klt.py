import cv2
import os
import numpy as np
import torch
import imageio.v3 as iio

# Load video frames dynamically using imageio
video_path = "./assets/rosbag_20.mp4"  # Update with the actual video file path
frames = iio.imiter(video_path, plugin="FFMPEG")

# Read the first frame
zeroth_frame = next(frames)
first_frame = next(frames)
gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)

# Define initial query points in the first frame (Modify as needed)
# query_points = np.array([[240, 93]], dtype=np.float32).reshape(-1, 1, 2)
# Select good feature points to track
N = 10  # Number of feature points to track
query_points = cv2.goodFeaturesToTrack(gray_prev, maxCorners=N, qualityLevel=0.01, minDistance=100)
query_points = query_points.astype(np.float32)  # Ensure correct type

# Number of query points (N)
N = query_points.shape[0]

print("query points shape:", query_points.shape)

# Initialize a list to store tracked points
tracked_points_list = [torch.tensor(query_points.squeeze(1))]

# Define Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,)
                #  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Track points through the video
prev_pts = query_points
S = 1  # Frame count

# Create a directory to save images
output_dir = "./saved_videos/tracked_images"
os.makedirs(output_dir, exist_ok=True)

for frame_idx, frame in enumerate(frames):
    S += 1  # Increment frame count
    gray_next = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Compute optical flow
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, prev_pts, None, **lk_params)

    print("status:", status)

    # Retain only successfully tracked points
    valid_pts = prev_pts.copy()
    valid_pts[status == 1] = next_pts[status == 1]

    # Store tracked points
    tracked_points_list.append(torch.tensor(valid_pts.squeeze(1)))

    # Visualize points on the current frame
    for pt in valid_pts:
        cv2.circle(frame, tuple(pt[0].astype(int)), 5, (0, 0, 255), -1)  # Red points

    # Save the frame with points plotted
    output_image_path = f"{output_dir}/frame_{frame_idx}.png"
    cv2.imwrite(output_image_path, frame)

    # Update for the next iteration
    gray_prev = gray_next
    prev_pts = valid_pts

    if S == 16:
        break

# Convert list to a PyTorch tensor with shape (S, N, 2)
tracked_points = torch.stack(tracked_points_list)

# Print results
print("Tracked Points Shape:", tracked_points.shape)
print(tracked_points)

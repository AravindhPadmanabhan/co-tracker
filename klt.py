import cv2
import os
import numpy as np
import torch
import imageio.v3 as iio

def track_klt(window_frames, query_points, pred_tracks):    
    first_frame = window_frames[0]
    gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    query_points = query_points.cpu().numpy().reshape(-1, 1, 2)

    # Create a directory to save images
    output_dir = "./saved_videos/tracked_images_klt"
    os.makedirs(output_dir, exist_ok=True)

    for pt in query_points:
        cv2.circle(window_frames[0], tuple(pt[0].astype(int)), 5, (0, 0, 255), -1)  # Red points
    output_image_path = f"{output_dir}/frame_0.png"
    cv2.imwrite(output_image_path, window_frames[0])

    # Initialize a list to store tracked points
    tracked_points_list = [torch.tensor(query_points.squeeze(1))]

    # Define Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,)
                    #  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Track points through the video
    prev_pts = query_points
    S = 1  # Frame count

    

    for frame_idx, frame in enumerate(window_frames[1:]):
        S += 1  # Increment frame count
        gray_next = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Compute optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, prev_pts, None, **lk_params)

        # Retain only successfully tracked points
        valid_pts = next_pts.copy()
        print("status: ", status)
        # valid_pts[status == 1] = next_pts[status == 1]

        # Store tracked points
        tracked_points_list.append(torch.tensor(valid_pts.squeeze(1)))

        # Visualize points on the current frame
        # for j in range(len(valid_pts)):
        # cv2.circle(frame, tuple(valid_pts[0][0].astype(int)), 1, (0, 0, 255), -1)  # Red points
        x1, y1 = valid_pts[0][0].astype(int)[0], valid_pts[0][0].astype(int)[1]
        x2, y2 = pred_tracks[0,frame_idx+1,0].cpu().int().tolist()
        # print("pixel value:", frame[x1, y1])
        # cv2.circle(frame, (x1,y1), 1, (0, 0, 255), -1)
        # cv2.circle(frame, (x2,y2), 1, (0, 255, 0), -1)
        frame[y1, x1] = [0, 0, 255]  # Red for (x1, y1)
        frame[y2, x2] = [0, 255, 0]  # Green for (x2, y2)
        print(x1, y1)

        # Save the frame with points plotted
        output_image_path = f"{output_dir}/frame_{frame_idx+1}.png"
        patch = frame[max(0,y1-25):y1+25, max(0,x1-25):x1+25]
        zoomed_patch = cv2.resize(patch, (50 * 10, 50 * 10), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_image_path, zoomed_patch)

        # Update for the next iteration
        gray_prev = gray_next
        prev_pts = valid_pts

        if S == 16:
            break

    # Convert list to a PyTorch tensor with shape (S, N, 2)
    tracked_points = torch.stack(tracked_points_list)
    return tracked_points


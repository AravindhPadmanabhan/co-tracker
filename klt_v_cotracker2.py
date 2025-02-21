# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
from klt import track_klt
import cv2

from cotracker.utils.visualizer import Visualizer
# from cotracker.predictor import CoTrackerOnlinePredictor
from window_tracker import CoTrackerWindow

from load_frames import load_images_from_folder


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# DEFAULT_DEVICE = ("cpu")
# print default device
print(f"Using device: {DEFAULT_DEVICE}")




if __name__ == "__main__":
    video_path="./assets/rosbag_20.mp4"
    model = CoTrackerWindow(checkpoint="checkpoints/scaled_online.pth", device="cuda")
    # model = model.to(DEFAULT_DEVICE)

    window_frames = []
    tracks_list = []

    queries = torch.tensor([[0, 401.0, 128.0]],
                            # [0, 200.0, 300.0],
                            # [0, 300.0, 400.0],
                            # [0, 300.0, 200.0],
                            # [0, 200.0, 200.0]], 
                            device=DEFAULT_DEVICE)
    removed_indices = [0]

    frames_generator = iio.imiter(video_path, plugin="FFMPEG")
    _ = next(frames_generator)  # skip the first frame
    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(frames_generator):
        if len(model.video) == 0:
            model.add_image(frame)
            forw_pts, status = model.track()
            window_frames.append(frame)
            continue
        else:
            if is_first_step:
                model.update_queries(queries[:,1:].cpu().tolist(), removed_indices)
                is_first_step = False
                removed_indices = []
            else:
                model.update_queries([[]], removed_indices)
            model.add_image(frame)
            tracks, status = model.track()
            tracks_list.append(tracks)
            
        window_frames.append(frame)
        if i == 15:
            break

    print("Tracks are computed")
    print("Tracks list:", tracks_list)
    print("WIndow frames length:", len(window_frames))

    # output_dir = "./saved_videos/tracked_images_cotracker"
    # os.makedirs(output_dir, exist_ok=True)
    # for i in range(len(window_frames)):
    #     for j in range(pred_tracks.shape[2]):
    #         x, y = pred_tracks[0,i,j].cpu().int().tolist()
    #         cv2.circle(window_frames[i], (x,y), 5, (0, 255, 0), -1)

    #     output_image_path = f"{output_dir}/frame_{i}.png"
    #     cv2.imwrite(output_image_path, window_frames[i])

    track_klt(window_frames, queries[:,1:], tracks_list[-1][:,-16:])

    

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
from cotracker.predictor import CoTrackerOnlinePredictor

from load_frames import load_images_from_folder


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# DEFAULT_DEVICE = ("cpu")
# print default device
print(f"Using device: {DEFAULT_DEVICE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/rosbag_20.mp4",
        help="path to a video",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    model = CoTrackerOnlinePredictor(checkpoint="checkpoints/scaled_online.pth", window_len=16)
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    queries = torch.tensor([[0, 401.0, 128.0]],
                            # [0, 200.0, 300.0],
                            # [0, 300.0, 400.0],
                            # [0, 300.0, 200.0],
                            # [0, 200.0, 200.0]], 
                            device=DEFAULT_DEVICE)

    def _process_step(window_frames, is_first_step):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), dtype=torch.float32, device=DEFAULT_DEVICE
            )
            .permute(0, 3, 1, 2)[None]
        ) 
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries = queries[None],
        )

    frames_generator = iio.imiter(args.video_path, plugin="FFMPEG")
    _ = next(frames_generator)  # skip the first frame
    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(frames_generator):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility, pred_confidence = _process_step(
                window_frames,
                is_first_step,
            )
            is_first_step = False
            if len(window_frames) == 16:
                break
        window_frames.append(frame)
        print("frame shape: ", frame.shape)
        print("frame datatype: ", frame.dtype)
        print("frame head: ", frame[:5,:5,:])

    print("Tracks are computed")

    # output_dir = "./saved_videos/tracked_images_cotracker"
    # os.makedirs(output_dir, exist_ok=True)
    # for i in range(len(window_frames)):
    #     for j in range(pred_tracks.shape[2]):
    #         x, y = pred_tracks[0,i,j].cpu().int().tolist()
    #         cv2.circle(window_frames[i], (x,y), 5, (0, 255, 0), -1)

    #     output_image_path = f"{output_dir}/frame_{i}.png"
    #     cv2.imwrite(output_image_path, window_frames[i])

    track_klt(window_frames, queries[:,1:], pred_tracks)

    

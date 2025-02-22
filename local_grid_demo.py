# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/MH_01_easy.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/scaled_online.pth",
        help="CoTracker model parameters",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")


    model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint, global_grid_size=5, local_grid_size=8, local_grid_extent=25)
    model = model.to(DEFAULT_DEVICE)

    window_frames = []
    queries = torch.tensor([[[0.0, 300.0, 60.0],
                             [0.0, 400.0, 120.0],
                             [0.0, 500.0, 180.0],
                             [0.0, 600.0, 240.0],
                             [0.0, 700.0, 300.0]]], device=DEFAULT_DEVICE)

    def _process_step(window_frames, is_first_step):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries,
        )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
            )
            is_first_step = False
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
    )

    print("Tracks are computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=0
    )

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
from cotracker.predictor_update import CoTrackerOnlinePredictor


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
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    model = CoTrackerOnlinePredictor(checkpoint="checkpoints/scaled_online.pth", window_len=16, local_grid_size=6, local_grid_extent=36)
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    queries = torch.tensor([[0, 700.0, 300.0], [0, 800.0, 400.0]], device=DEFAULT_DEVICE)
    removed_indices = []
    new_queries_num = 0

    queries1 = torch.tensor([[0, 800.0, 400.0], [19, 600.0, 400.0]], device=DEFAULT_DEVICE)
    removed_indices1 = [0]
    new_queries_num1 = 1

    def _process_step(window_frames, is_first_step, input_queries, input_removed_indices, input_new_queries_num):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-9 :]), dtype=torch.float32, device=DEFAULT_DEVICE
            )
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        # print("Video chunk shape: ", video_chunk.shape)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries = input_queries[None],
            removed_indices = input_removed_indices,
            new_queries_num = input_new_queries_num,
        )

    tracks = torch.zeros(1, 0, 2, 2, device=DEFAULT_DEVICE)
    visibility = torch.zeros(1, 0, 2, device=DEFAULT_DEVICE)

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        if i > 8:
            if i == 21:  # Frames 0-20 in window_frames
                pred_tracks, pred_visibility, pred_confidence = _process_step(
                    window_frames,
                    is_first_step,
                    queries1,
                    removed_indices1,
                    new_queries_num1,
                )
            elif i > 21:
                pred_tracks, pred_visibility, pred_confidence = _process_step(
                    window_frames,
                    is_first_step,
                    queries1,
                    removed_indices,
                    new_queries_num,
                )
            else:
                pred_tracks, pred_visibility, pred_confidence = _process_step(
                    window_frames,
                    is_first_step,
                    queries,
                    removed_indices,
                    new_queries_num,
                )

            if is_first_step:
                tracks = pred_tracks
                visibility = pred_visibility
            else:
                tracks = torch.cat([tracks, pred_tracks[:,-1].unsqueeze(1)], dim=1)
                visibility = torch.cat([visibility, pred_visibility[:,-1].unsqueeze(1)], dim=1)

            is_first_step = False
            # print("Window length: ", len(window_frames))
            if len(window_frames) == 40:
                break

        window_frames.append(frame)

    print("Tracks are computed")
    # print(pred_confidence)

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), dtype=torch.float32, device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, tracks, visibility, #query_frame=args.grid_query_frame
    )

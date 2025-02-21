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

    model = CoTrackerOnlinePredictor(checkpoint="checkpoints/scaled_online.pth", window_len=16)
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    queries = [torch.tensor([[0, 200.0, 200.0], [0, 100.0, 600.0]], device=DEFAULT_DEVICE),
               torch.tensor([[0, 200.0, 200.0], [0, 100.0, 600.0]], device=DEFAULT_DEVICE),
               torch.tensor([[0, 200.0, 200.0], [16, 150.0, 250.0]], device=DEFAULT_DEVICE),
               torch.tensor([[0, 200.0, 200.0], [16, 150.0, 250.0]], device=DEFAULT_DEVICE),
               torch.tensor([[0, 200.0, 200.0], [32, 600.0, 300.0]], device=DEFAULT_DEVICE),
               torch.tensor([[0, 200.0, 200.0], [32, 600.0, 300.0]], device=DEFAULT_DEVICE)]
    removed_indices = [[], [], [1], [], [1], []]

    def _process_step(window_frames, is_first_step, input_queries, input_removed_indices):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), dtype=torch.float32, device=DEFAULT_DEVICE
            )
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        # print("Video chunk shape: ", video_chunk.shape)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries = input_queries[None],
            removed_indices = input_removed_indices,
        )
    
    tracks = None
    visibility = None

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility, pred_confidence = _process_step(
                window_frames,
                is_first_step,
                queries[i//model.step - 1],
                removed_indices[i//model.step - 1],
            )
            is_first_step = False
            # print("Window length: ", len(window_frames))
            if tracks is None:
                tracks = pred_tracks
                visibility = pred_visibility
            else:
                tracks = torch.cat((tracks, pred_tracks[:,-8:]), dim=1)
                visibility = torch.cat((visibility, pred_visibility[:,-8:]), dim=1)
            if len(window_frames) == 48:
                break

        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    # pred_tracks, pred_visibility, pred_confidence = _process_step(
    #     window_frames[-(i % model.step) - model.step - 1 :],
    #     is_first_step,
    #     grid_size=args.grid_size,
    #     grid_query_frame=args.grid_query_frame,
    # )

    # tracks = torch.cat((tracks, pred_tracks[:,-8:]), dim=1)

    print("Tracks are computed")
    # print(pred_confidence)

    print("Tracks shape: ", tracks.shape)
    print("Visibility shape: ", pred_visibility.shape)
    print("Video len: ", len(window_frames))
    # print(tracks)

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), dtype=torch.float32, device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, tracks, visibility, #query_frame=args.grid_query_frame
    )

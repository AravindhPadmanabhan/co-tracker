import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np

from cotracker.utils.visualizer import Visualizer
from onnx_export2 import CoTrackerWrapper
from onnx_export3 import CoTrackerOnlineWrapper
from onnx_demo import get_queries, onnx_inference

use_onnx = False

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)


def _process_step(window_frames, is_first_step):
    video_chunk = (
        torch.tensor(
            np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
        )
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)
    queries = torch.zeros(1, 100, 3)
    if is_first_step:
        queries = get_queries(video_chunk, grid_size=10)

    if use_onnx:
        return onnx_inference("./cotracker_online.onnx", video_chunk, queries)
    return model.forward(
        video_chunk,
        queries=queries,
    )

if __name__ == "__main__":
    checkpoint = "./checkpoints/scaled_online.pth"
    grid_size = 10
    video_path = "./assets/apple.mp4"
    grid_query_frame = 0

    model = CoTrackerOnlineWrapper(checkpoint).to(DEFAULT_DEVICE)

    window_frames = []

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(
            video_path,
            plugin="FFMPEG",
        )
    ):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step
            )
            is_first_step = False
        window_frames.append(frame)


    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step
    )

    print("Tracks are computed")
    # print("pred_tracks shape:", pred_tracks.shape)

    # save a video with predicted tracks
    seq_name = video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=grid_query_frame
    )

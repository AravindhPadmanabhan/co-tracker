import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np

from cotracker.utils.visualizer import Visualizer
from onnx_export2 import CoTrackerWrapper
from onnx_export4 import CoTrackerOnlineWrapper
from onnx_demo import get_queries, onnx_inference

use_onnx = False
grid_size = 10

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class WrapperDemo:
    def __init__(self, model, grid_size, device):
        self.model = model
        self.grid_size = grid_size
        self.device = device
        self.queries = None

    def process_step(self, window_frames, is_first_step, *args):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-self.model.step * 2 :]), device=self.device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        if is_first_step:
            self.queries = get_queries(video_chunk, grid_size=self.grid_size)
            return (
                torch.tensor(torch.nan).expand(1, self.model.step, self.queries.shape[1], 2).to(video_chunk.device),
                torch.tensor(torch.nan).expand(1, self.model.step, self.queries.shape[1]).to(video_chunk.device),
                *args
            )

        if use_onnx:
            return onnx_inference("./cotracker_online.onnx", video_chunk, self.queries)

        return self.model.forward(
            video_chunk, self.queries, *args
        )





if __name__ == "__main__":
    checkpoint = "./checkpoints/scaled_online.pth"
    video_path = "./assets/apple.mp4"
    grid_query_frame = 0

    model = CoTrackerOnlineWrapper(checkpoint).to(DEFAULT_DEVICE)

    wrapper_demo = WrapperDemo(model, grid_size, DEFAULT_DEVICE)

    online_ind = torch.zeros(1).to(DEFAULT_DEVICE)
    online_track_feat = torch.tensor(torch.nan).expand(model.corr_levels,1,grid_size*grid_size,model.latent_dim).to(DEFAULT_DEVICE)
    online_track_support = torch.tensor(torch.nan).expand(model.corr_levels,(2*model.corr_radius + 1)**2,grid_size*grid_size,model.latent_dim).to(DEFAULT_DEVICE)
    online_coords_predicted = torch.tensor(torch.nan).expand(1,model.step,grid_size*grid_size,2).to(DEFAULT_DEVICE)
    online_vis_predicted = torch.tensor(torch.nan).expand(1,model.step,grid_size*grid_size).to(DEFAULT_DEVICE)
    online_conf_predicted = torch.tensor(torch.nan).expand(1,model.step,grid_size*grid_size).to(DEFAULT_DEVICE)

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
            results = wrapper_demo._process_step(
                window_frames,
                is_first_step,
                online_ind,
                online_track_feat,
                online_track_support,
                online_coords_predicted,
                online_vis_predicted,
                online_conf_predicted,
            )
            is_first_step = False

            online_ind = results[2]
            online_track_feat = results[3]
            online_track_support = results[4]
            online_coords_predicted = results[5]
            online_vis_predicted = results[6]
            online_conf_predicted = results[7]
        
        window_frames.append(frame)


    # Processing the final video frames in case video length is not a multiple of model.step
    results = wrapper_demo._process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        online_ind,
        online_track_feat,
        online_track_support,
        online_coords_predicted,
        online_vis_predicted,
        online_conf_predicted,
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
        video, results[0], results[1], query_frame=grid_query_frame
    )

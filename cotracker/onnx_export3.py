import torch
import torch.nn as nn
import torch.nn.functional as F
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = "./checkpoints/scaled_online.pth"

class CoTrackerOnlineWrapper(nn.Module):
    def __init__(
        self,
        checkpoint="./checkpoints/scaled_online.pth",
        window_len=16,
    ):
        super().__init__()
        model = CoTrackerThreeOnline(stride=4, corr_radius=3, window_len=window_len)
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location="cpu")
                if "model" in state_dict:
                    state_dict = state_dict["model"]
            model.load_state_dict(state_dict)
        self.interp_shape = model.model_resolution
        self.step = model.window_len // 2
        self.model = model
        self.first_step = True
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video_chunk,
        queries: torch.Tensor = None,
    ):
        B, T, C, H, W = video_chunk.shape
        if self.first_step:
            self.model.init_video_online_processing()         
            self.queries = queries
            self.first_step = False
            # return (torch.zeros(B,T,queries.shape[1],2).to(video_chunk.device), torch.zeros(B,T,queries.shape[1]).to(video_chunk.device))
            return (None, None)

        video_chunk = video_chunk.reshape(B * T, C, H, W)
        video_chunk = F.interpolate(
            video_chunk, tuple(self.interp_shape), mode="bilinear", align_corners=True
        )
        video_chunk = video_chunk.reshape(
            B, T, 3, self.interp_shape[0], self.interp_shape[1]
        )

        tracks, visibilities, confidence, __ = self.model(
            video=video_chunk, queries=self.queries, iters=6, is_online=True
        )
            
        visibilities = visibilities * confidence
        thr = 0.6
        return (
            tracks
            * tracks.new_tensor(
                [
                    (W - 1) / (self.interp_shape[1] - 1),
                    (H - 1) / (self.interp_shape[0] - 1),
                ]
            ),
            visibilities > thr,
        )


if __name__ == "__main__":
    # Create an instance of the wrapper
    wrapper_model = CoTrackerOnlineWrapper(checkpoint).to(device)

    # Create dummy input
    B = 1  # Batch size
    T = 10  # Number of frames (time steps)
    C = 3  # Number of channels (RGB)
    H = 576  # Height
    W = 640  # Width
    dummy_video = torch.randn(B, T, C, H, W).float().to(device)
    dummy_queries =  torch.randn(B, 100, 3).float().to(device)
    first_step=True

    # Export the model to ONNX format
    torch.onnx.export(
        wrapper_model,
        args=(dummy_video,dummy_queries, first_step),
        f="cotracker_online.onnx",
        input_names=['video', 'queries', 'first_step'],
        output_names=['pred_tracks', 'pred_visibility'],
        dynamic_axes={
            'video': {1: 'time'},  # Allow variable sequence length
            'pred_tracks': {1: 'time', 2: 'num_tracks'},
            'pred_visibility': {1: 'time', 2: 'num_tracks'}
        },
        opset_version=20
    )
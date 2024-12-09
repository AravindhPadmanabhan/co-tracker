import torch
import torch.nn as nn
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = "./checkpoints/scaled_online.pth"


# Fixed grid_size
grid_size = 10

# Wrapper is needed for the grid_size
class CoTrackerWrapper(CoTrackerOnlinePredictor):
    def __init__(self, checkpoint):
        super(CoTrackerWrapper, self).__init__(checkpoint=checkpoint, offline=False, window_len=16)
        # self.first_step = True

    def forward(self, video, queries, is_first_step):
        # if self.first_step:
        #     self.queries = queries
        #     self.N = queries.shape[1]
        #     self.first_step = False
        return super(CoTrackerWrapper, self).forward(video, is_first_step=is_first_step, queries=queries, grid_size=10)

if __name__ == "__main__":
    # Create an instance of the wrapper
    wrapper_model = CoTrackerWrapper(checkpoint).to(device)

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
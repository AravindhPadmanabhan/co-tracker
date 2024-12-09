import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the CoTracker model
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

# Fixed grid_size
grid_size = 10

# Wrapper is needed for the grid_size
class CoTrackerWrapper(nn.Module):
    def __init__(self, model):
        super(CoTrackerWrapper, self).__init__()
        self.model = model
        self.grid_size = grid_size

    def forward(self, video):
        return self.model(video, grid_size=self.grid_size)


# Create an instance of the wrapper
wrapper_model = CoTrackerWrapper(cotracker).to(device)

# Create dummy input
B = 1  # Batch size
T = 10  # Number of frames (time steps)
C = 3  # Number of channels (RGB)
H = 576  # Height
W = 640  # Width
dummy_video = torch.randn(B, T, C, H, W).float().to(device)

# Export the model to ONNX format
torch.onnx.export(
    wrapper_model,
    dummy_video,
    "cotracker.onnx",
    input_names=['video'],
    output_names=['pred_tracks', 'pred_visibility'],
    dynamic_axes={
        'video': {1: 'time'},  # Allow variable sequence length
        'pred_tracks': {1: 'time', 2: 'num_tracks'},
        'pred_visibility': {1: 'time', 2: 'num_tracks'}
    },
    opset_version=16
)
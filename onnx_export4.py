import torch
import torch.nn as nn
import torch.nn.functional as F
from cotracker.models.core.cotracker.cotracker3_export4 import CoTrackerThreeExport

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = "./checkpoints/scaled_online.pth"

class CoTrackerOnlineWrapper(nn.Module):
    def __init__(
        self,
        checkpoint="./checkpoints/scaled_online.pth"
    ):
        super().__init__()
        model = CoTrackerThreeExport(stride=4, corr_radius=3, window_len=16)
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location="cpu", weights_only=True)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
            model.load_state_dict(state_dict)
        self.interp_shape = model.model_resolution
        self.step = model.window_len // 2
        self.model = model
        self.model.eval()

    def preprocess_inputs(self, online_ind, online_track_feat, online_track_support, online_coords_predicted, online_vis_predicted, online_conf_predicted):
        online_ind = int(online_ind.item())
        if torch.isnan(online_track_feat).any().item():
            online_track_feat = [None]*self.model.corr_levels
        else:
            online_track_feat = [online_track_feat[i] for i in range(online_track_feat.shape[0])]
        if torch.isnan(online_track_support).any().item():
            online_track_support = [None]*self.model.corr_levels
        else:
            online_track_support = [online_track_support[i] for i in range(online_track_support.shape[0])] 
        if torch.isnan(online_coords_predicted).any().item():
            online_coords_predicted = None
        if torch.isnan(online_vis_predicted).any().item():
            online_vis_predicted = None
        if torch.isnan(online_conf_predicted).any().item():
            online_conf_predicted = None
        return online_ind, online_track_feat, online_track_support, online_coords_predicted, online_vis_predicted, online_conf_predicted


    def postprocess_results(self, results):
        results[3] = torch.tensor(results[3]).to(device)
        results[4] = torch.cat(results[4], dim=0).to(device)
        results[5] = torch.cat(results[5], dim=0).to(device)
        return results

    @torch.no_grad()
    def forward(
        self,
        video_chunk,
        queries: torch.Tensor = None,
        online_ind=0,
        online_track_feat=None,
        online_track_support=None,
        online_coords_predicted=None,
        online_vis_predicted=None,
        online_conf_predicted=None,
    ):
        B, T, C, H, W = video_chunk.shape
        video_chunk = video_chunk.reshape(B * T, C, H, W)
        video_chunk = F.interpolate(
            video_chunk, tuple(self.interp_shape), mode="bilinear", align_corners=True
        )
        video_chunk = video_chunk.reshape(
            B, T, 3, self.interp_shape[0], self.interp_shape[1]
        )

        # online_ind, online_track_feat, online_track_support, online_coords_predicted, online_vis_predicted, online_conf_predicted = self.preprocess_inputs(
        #     online_ind, online_track_feat, online_track_support, online_coords_predicted, online_vis_predicted, online_conf_predicted
        # )

        results = self.model(
            video_chunk, queries, 
            online_ind, 
            online_track_feat, 
            online_track_support, 
            online_coords_predicted, 
            online_vis_predicted, 
            online_conf_predicted,
            iters=6
        )

        results = list(results)
        # results = self.postprocess_results(results)
            
        results[1] = results[1] * results[2]
        thr = 0.6
        return (
            results[0],
            results[1] > thr,
            results[3],
            results[4],
            results[5],
            results[6],
            results[7],
            results[8],
        )


if __name__ == "__main__":
    # Create an instance of the wrapper
    wrapper_model = CoTrackerOnlineWrapper(checkpoint).to(device)

    wrapper_model.eval()

    # for module in wrapper_model.modules():
    #     if isinstance(module, nn.InstanceNorm2d):
    #         module.track_running_stats = True
    #         module.running_var = None
    #         module.running_mean = None

    # Create dummy input
    B = 1  # Batch size
    T = 16  # Number of frames (time steps)
    C = 3  # Number of channels (RGB)
    H = 720  # Height
    W = 1296  # Width
    dummy_video = torch.randn(B, T, C, H, W).float().to(device)
    dummy_queries =  torch.randn(B, 100, 3).float().to(device)
    ind = torch.tensor([0]).to(device)
    track_feat = torch.randn(4,1,1,100,128).to(device)
    track_support = torch.randn(4,1,49,100,128).to(device)
    coords_predicted = torch.zeros(1,8,100,2).to(device)
    vis_predicted = torch.zeros(1,8,100).to(device)
    conf_predicted = torch.zeros(1,8,100).to(device)

    # Export the model to ONNX format
    torch.onnx.export(
        wrapper_model,
        args=(dummy_video,dummy_queries, ind, track_feat, track_support, coords_predicted, vis_predicted, conf_predicted),
        f="cotracker_wrapper.onnx",
        input_names=['video', 'queries', 'ind', 'track_feat', 'track_support', 'coords_predicted', 'vis_predicted', 'conf_predicted'],
        output_names=['pred_tracks', 'pred_visibility', 'o_ind', 'o_track_feat', 'o_track_support', 'o_coords_predicted', 'o_vis_predicted', 'o_conf_predicted'],
        dynamic_axes={
            'video': {1: 'time'},  # Allow variable sequence length
            'queries': {1: 'num_tracks'},
            'ind': {},
            'track_feat': {1: 'num_tracks'},
            'track_support': {1: 'num_tracks'},
            'coords_predicted': {1: 'time', 2: 'num_tracks'},
            'vis_predicted': {1: 'time', 2: 'num_tracks'},
            'conf_predicted': {1: 'time', 2: 'num_tracks'},
            'pred_tracks': {1: 'time', 2: 'num_tracks'},
            'pred_visibility': {1: 'time', 2: 'num_tracks'},
            'o_ind': {},
            'o_track_feat': {1: 'num_tracks'},
            'o_track_support': {1: 'num_tracks'},
            'o_coords_predicted': {1: 'time', 2: 'num_tracks'},
            'o_vis_predicted': {1: 'time', 2: 'num_tracks'},
            'o_conf_predicted': {1: 'time', 2: 'num_tracks'}
        },
        opset_version=20,
        # do_constant_folding=False,
    )
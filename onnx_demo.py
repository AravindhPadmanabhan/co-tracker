import onnxruntime as ort
import torch
from cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid

def onnx_inference(model_path, video, queries, ind, track_feat, track_support, coords_predicted, vis_predicted, conf_predicted):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Get input and output details
    input_names = [inp.name for inp in session.get_inputs()]
    input_shapes = [inp.shape for inp in session.get_inputs()]
    input_types = [inp.type for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    print(f"Input Name: {input_names}, Shape: {input_shapes}, Type: {input_types}")
    print(f"Output Name: {output_names}")

    # Prepare sample inputs as PyTorch tensors (modify shapes as per your model's input requirements)
    input_data = {
        input_names[0]: video.cpu().numpy(),   # Convert video to NumPy
        input_names[1]: queries.cpu().numpy(),  # Convert queries to NumPy
        input_names[2]: ind.cpu().numpy(),
        input_names[3]: track_feat.cpu().numpy(),
        input_names[4]: track_support.cpu().numpy(),
        input_names[5]: coords_predicted.cpu().numpy(),
        input_names[6]: vis_predicted.cpu().numpy(),
        input_names[7]: conf_predicted.cpu().numpy(),

    }

    # Run inference
    outputs = session.run(output_names, input_data)
    
    # Convert the outputs (NumPy arrays) back to PyTorch tensors
    output_tensors = [torch.tensor(output) for output in outputs]
    print("Pred tracks shape: ", output_tensors[0].shape)

    return output_tensors

def get_queries(video_chunk, grid_size, interp_shape=(384, 512)):
    grid_pts = get_points_on_a_grid(grid_size, interp_shape, device=video_chunk.device)
    N = grid_size**2
    queries = torch.cat(
        [torch.ones_like(grid_pts[:, :, :1]) * 0, grid_pts],
        dim=2,
    ).to(video_chunk.device)
    return queries

#         coords_predicted_padded = F.pad(
#             online_coords_predicted, (0, 0, 0, 0, 0, pad.item()), "constant"
#         )
#         vis_predicted_padded = F.pad(
#             online_vis_predicted, (0, 0, 0, pad.item()), "constant"
#         )
#         conf_predicted_padded = F.pad(
#             online_conf_predicted, (0, 0, 0, pad.item()), "constant"
#         )

#         # Logic for selecting fallback or padded predictions
#         coords_predicted = torch.where(
#             nan_mask, coords_predicted, coords_predicted_padded
#         )
#         vis_predicted = torch.where(
#             nan_mask, vis_predicted, vis_predicted_padded
#         )
#         conf_predicted = torch.where(
#             nan_mask, conf_predicted, conf_predicted_padded
#         )

#         online_coords_predicted = torch.where(
#             nan_mask, coords_predicted, online_coords_predicted
#         )
#         online_vis_predicted = torch.where(
#             nan_mask, vis_predicted, online_vis_predicted
#         )
#         online_conf_predicted = torch.where(
#             nan_mask, conf_predicted, online_conf_predicted
#         )
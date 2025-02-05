# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeBase, posenc

torch.manual_seed(0)


class CoTrackerThreeExport(CoTrackerThreeBase):
    def __init__(self, **args):
        super(CoTrackerThreeExport, self).__init__(**args)

    def forward_window(
        self,
        fmaps_pyramid,
        coords,
        track_feat_support_pyramid,
        vis=None,
        conf=None,
        attention_mask=None,
        iters=4,
        add_space_attn=False,
    ):
        B, S, *_ = fmaps_pyramid[0].shape
        N = coords.shape[2]
        r = 2 * self.corr_radius + 1

        coord_preds, vis_preds, conf_preds = [], [], []
        for it in range(iters):
            coords = coords.detach()  # B T N 2
            coords_init = coords.view(B * S, N, 2)
            corr_embs = []
            corr_feats = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_emb = self.corr_mlp(corr_volume.reshape(B * S * N, r * r * r * r))

                corr_embs.append(corr_emb)

            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.view(B, S, N, corr_embs.shape[-1])

            transformer_input = [vis, conf, corr_embs]

            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )

            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=coords.device,
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )  # batch, num_points, num_frames, 84
            transformer_input.append(rel_pos_emb_input)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, S, -1)
            )

            x = x + self.interpolate_time_embed(x, S)
            x = x.view(B, N, S, -1)  # (B N) T D -> B N T D

            delta = self.updateformer(x, add_space_attn=add_space_attn)

            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
            delta_conf = delta[..., 3:].permute(0, 2, 1, 3)

            vis = vis + delta_vis
            conf = conf + delta_conf

            coords = coords + delta_coords
            coord_preds.append(coords[..., :2] * float(self.stride))

            vis_preds.append(vis[..., 0])
            conf_preds.append(conf[..., 0])
        return coord_preds, vis_preds, conf_preds

    def forward(
        self,
        video,
        queries,
        online_ind=0,
        online_track_feat=None,
        online_track_support=None,
        online_coords_predicted=None,
        online_vis_predicted=None,
        online_conf_predicted=None,
        iters=4,
        add_space_attn=True,
        fmaps_chunk_size=200,
    ):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """

        B, T, C, H, W = video.shape
        device = queries.device
        # assert H % self.stride == 0 and W % self.stride == 0

        B, N, __ = queries.shape
        # B = batch size
        # S_trimmed = actual number of frames in the window
        # N = number of tracks
        # C = color channels (3 for RGB)
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # video = B T C H W
        # queries = B N 3
        # coords_init = B T N 2
        # vis_init = B T N 1
        S = self.window_len

        step = S // 2  # How much the sliding window moves at every step

        video = 2 * (video / 255.0) - 1.0
        pad = (S - T)
        video = video.reshape(B, 1, T, C * H * W)
        if pad > 0:
            padding_tensor = video[:, :, -1:, :].expand(B, 1, pad, C * H * W)
            video = torch.cat([video, padding_tensor], dim=2)
        video = video.reshape(B, -1, C, H, W)
        T_pad = video.shape[1]
        # The first channel is the frame number
        # The rest are the coordinates of points we want to track
        dtype = video.dtype
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride

        # We store our predictions here
        coords_predicted = torch.zeros((B, T, N, 2), device=device)
        vis_predicted = torch.zeros((B, T, N), device=device)
        conf_predicted = torch.zeros((B, T, N), device=device)

        if online_coords_predicted is None:
            # Init online predictions with zeros
            online_coords_predicted = coords_predicted
            online_vis_predicted = vis_predicted
            online_conf_predicted = conf_predicted
        else:
            # Pad online predictions with zeros for the current window
            pad = min(step, T - step)
            coords_predicted = F.pad(
                online_coords_predicted, (0, 0, 0, 0, 0, pad), "constant"
            )
            vis_predicted = F.pad(
                online_vis_predicted, (0, 0, 0, pad), "constant"
            )
            conf_predicted = F.pad(
                online_conf_predicted, (0, 0, 0, pad), "constant"
            )

        C_ = C
        H4, W4 = H // self.stride, W // self.stride

        # Compute convolutional features for the video or for the current chunk in case of online mode
        if T > fmaps_chunk_size:
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = video[:, t : t + fmaps_chunk_size]
                fmaps_chunk = self.fnet(video_chunk.reshape(-1, C_, H, W))
                T_chunk = video_chunk.shape[1]
                C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                fmaps.append(fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk))
            fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
        else:
            fmaps = self.fnet(video.reshape(-1, C_, H, W))
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        fmaps = fmaps.to(dtype)

        # We compute track features
        fmaps_pyramid = []
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T_pad, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T_pad, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        sample_frames = queried_frames[:, None, :, None]  # B 1 N 1
        left = 0 if online_ind == 0 else online_ind + step
        right = online_ind + S
        sample_mask = (sample_frames >= left) & (sample_frames < right)

        for i in range(self.corr_levels):
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames - online_ind,
                queried_coords / 2**i,
                support_radius=self.corr_radius,
            )

            if online_track_feat[i] is None:
                online_track_feat[i] = torch.zeros_like(
                    track_feat, device=device
                )
                online_track_support[i] = torch.zeros_like(
                    track_feat_support, device=device
                )
            online_track_feat[i] += track_feat * sample_mask
            online_track_support[i] += track_feat_support * sample_mask
            track_feat_pyramid.append(
                online_track_feat[i].repeat(1, T_pad, 1, 1)
            )
            track_feat_support_pyramid.append(
                online_track_support[i].unsqueeze(1)
            )

        D_coords = 2
        coord_preds, vis_preds, confidence_preds = [], [], []

        vis_init = torch.zeros((B, S, N, 1), device=device).float()
        conf_init = torch.zeros((B, S, N, 1), device=device).float()
        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, S, N, 2).float()

        num_windows = (T - S + step - 1) // step + 1
        # We process only the current video chunk in the online mode
        indices = [online_ind]

        for ind in indices:
            if ind > 0:
                overlap = S - step
                copy_over = (queried_frames < ind + overlap)[
                    :, None, :, None
                ]  # B 1 N 1
                coords_prev = coords_predicted[:, ind : ind + overlap] / self.stride
                padding_tensor = coords_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                coords_prev = torch.cat([coords_prev, padding_tensor], dim=1)

                vis_prev = vis_predicted[:, ind : ind + overlap, :, None].clone()
                padding_tensor = vis_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                vis_prev = torch.cat([vis_prev, padding_tensor], dim=1)

                conf_prev = conf_predicted[:, ind : ind + overlap, :, None].clone()
                padding_tensor = conf_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                conf_prev = torch.cat([conf_prev, padding_tensor], dim=1)

                coords_init = torch.where(
                    copy_over.expand_as(coords_init), coords_prev, coords_init
                )
                vis_init = torch.where(
                    copy_over.expand_as(vis_init), vis_prev, vis_init
                )
                conf_init = torch.where(
                    copy_over.expand_as(conf_init), conf_prev, conf_init
                )

            attention_mask = (queried_frames < ind + S).reshape(B, 1, N)  # B S N
            # import ipdb; ipdb.set_trace()
            coords, viss, confs = self.forward_window(
                fmaps_pyramid=(
                    fmaps_pyramid
                ),
                coords=coords_init,
                track_feat_support_pyramid=[
                    attention_mask[:, None, :, :, None] * tfeat
                    for tfeat in track_feat_support_pyramid
                ],
                vis=vis_init,
                conf=conf_init,
                attention_mask=attention_mask.repeat(1, S, 1),
                iters=iters,
                add_space_attn=add_space_attn,
            )
            S_trimmed = (T)  # accounts for last window duration
            coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed]
            vis_predicted[:, ind : ind + S] = viss[-1][:, :S_trimmed]
            conf_predicted[:, ind : ind + S] = confs[-1][:, :S_trimmed]
                
        online_ind += step
        online_coords_predicted = coords_predicted
        online_vis_predicted = vis_predicted
        online_conf_predicted = conf_predicted
        vis_predicted = torch.sigmoid(vis_predicted)
        conf_predicted = torch.sigmoid(conf_predicted)

        return coords_predicted, vis_predicted, conf_predicted, online_ind, online_track_feat, online_track_support, online_coords_predicted, online_vis_predicted, online_conf_predicted

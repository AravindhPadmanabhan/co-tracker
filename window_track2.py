from cv_bridge import CvBridge
from cotracker.predictor import CoTrackerOnlinePredictor
import torch
import numpy as np
import cv2

class CoTrackerWindow:
    def __init__(self, checkpoint, device='cuda'):
        self.model = CoTrackerOnlinePredictor(checkpoint=checkpoint)
        self.model.to(device)
        self.video = []
        self.frame_numbers = []
        self.frame_no = 1
        self.video_len = self.model.model.window_len + self.model.step
        # self.video_padded = None
        self.max_queries = 100
        self.queries = None
        self.cur_tracks = None
        self.track_status = torch.ones(self.max_queries, dtype=torch.bool).to(device)
        self.new_queries = None
        self.device = device

    def add_image(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        # print(img_tensor)
        self.video.append(img_tensor)
        self.frame_numbers.append(self.frame_no)
        if len(self.video) == 1:
            self.video = self.video + [self.video[-1]] * (self.video_len - len(self.video))
            self.frame_numbers = self.frame_numbers + [self.frame_numbers[-1]] * (self.video_len - len(self.frame_numbers))
        if len(self.video) > self.video_len:
            self.video.pop(0)
            self.frame_numbers.pop(0)
        self.frame_no += 1

    def update_queries(self, new_points, removed_indices):
        if len(removed_indices) > 0:
            new_points = torch.tensor(new_points, dtype=torch.float32).to(self.device)  # Shape: (N,2)
            frame = torch.ones(new_points.shape[0], 1).to(self.device) * (self.video_len - 1)  # Shape: (N,1)
            self.new_queries = torch.cat((frame, new_points), dim=1).unsqueeze(0)  # Shape: (1,N,3)
            if self.queries is None:
                self.queries = self.new_queries
                self.max_queries = self.queries.shape[1]
            else:
                mask = torch.ones(self.queries.shape[1], dtype=torch.bool)  # Create a mask for all points
                mask[removed_indices] = False
                self.queries = self.queries[:, mask, :]
                self.queries = torch.cat((self.queries, self.new_queries), dim=1)
        else:
            self.new_queries = torch.zeros(1,0,3).to(self.device)
        
        # Move queries behind by one frame:
        self.queries[0, :, 0] -= 1
        if self.cur_tracks is not None:
            out_of_window_mask = self.queries[0, :, 0] < 0
            traces = torch.cat((torch.zeros(1,self.max_queries,1).to(self.device), self.cur_tracks[:,1,:,:]), dim=-1)
            self.queries = torch.where(out_of_window_mask.unsqueeze(-1), traces, self.queries)

    def track(self):
        
        print("query: ", self.queries)

        if self.queries is None:
            return None, None

        is_first_step = True
        for i in range(3):
            video_chunk = self.video[:(i+1)*self.model.step]
            window_frames = video_chunk[-self.model.model.window_len:]
            window_frames = torch.stack(window_frames).to(self.device).unsqueeze(0)
            tracks, vis, conf = self.model(window_frames, is_first_step=is_first_step, queries=self.queries)
            if is_first_step:
                is_first_step = False

        self.track_status = ((conf[0,-1,:] > 0.6) * vis[0,-1,:]) == 1
        self.cur_tracks = tracks

        return tracks, self.track_status
from cotracker.predictor import CoTrackerOnlinePredictor
import torch
import numpy as np
import cv2

import imageio.v3 as iio
import numpy as np

class CoTrackerWindow:
    def __init__(self, checkpoint, device='cuda'):
        self.model = CoTrackerOnlinePredictor(checkpoint=checkpoint)
        self.model.to(device)
        self.video = []
        self.video_len = self.model.model.window_len + self.model.step
        # self.video_padded = None
        self.max_queries = 100
        self.sift = cv2.SIFT_create(nfeatures=self.max_queries)
        self.queries = None
        self.device = device
        

    def add_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)
        self.video.append(img_tensor)
        if len(self.video) == 1:
            self.video = self.video + [self.video[-1]] * (self.video_len - len(self.video))
        if len(self.video) > self.video_len:
            self.video.pop(0)
        
        # self.pad_video()

    # def pad_video(self):
    #     if len(self.video) < self.video_len:
    #         self.video_padded = self.video + [self.video[-1]] * (self.video_len - len(self.video))
    #     else:
    #         self.video_padded = self.video

    def get_queries(self):
        query_frame = self.video[0]

        # print(query_frame.shape)

        image_np = query_frame.permute(1, 2, 0).cpu().numpy()
        image_np = image_np.astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

        # gray_image = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)
        

        keypoints = self.sift.detect(gray_image, None)
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)  # Sort by response

        keypoints = keypoints[:self.max_queries]

        # Convert keypoints to a list of (x, y) coordinates
        keypoints_xy = [kp.pt for kp in keypoints]  # Extract x, y coordinates

        # Convert to PyTorch tensors
        keypoints_tensor = torch.tensor(keypoints_xy, dtype=torch.float32).to(self.device)  # Shape: (N, 2)

        print(keypoints_tensor.shape)
        
        self.queries = torch.cat((torch.zeros(self.max_queries, 1).to(self.device), keypoints_tensor), dim=1)  # Shape: (N, 3)
        self.queries = self.queries.unsqueeze(0) # Shape: (1, N, 3)

    def track(self):
        # if self.queries is None:
        self.get_queries()

        is_first_step = True
        for i in range(3):
            video_chunk = self.video[:(i+1)*self.model.step]
            window_frames = video_chunk[-self.model.model.window_len:]
            window_frames = torch.stack(window_frames).to(self.device).unsqueeze(0)
            pred_tracks, pred_visibility = self.model(window_frames, is_first_step=is_first_step, queries=self.queries)
            if is_first_step:
                is_first_step = False

        return pred_tracks, pred_visibility
    
    def debug_features(self):
        # Mark the queries on the image
        query_frame = self.video[0]
        query_frame_np = query_frame.permute(1, 2, 0).cpu().numpy()
        query_frame_np = cv2.cvtColor(query_frame_np, cv2.COLOR_RGB2BGR)
        for query in self.queries[0]:
            x, y = int(query[1]), int(query[2])
            cv2.circle(query_frame_np, (x, y), 5, (0, 255, 0), -1)

        return query_frame_np
        

if __name__ == '__main__':
    # Load the model
    model = CoTrackerWindow(checkpoint='./checkpoints/scaled_online.pth')

    for i, frame in enumerate(
        iio.imiter(
            "./assets/apple.mp4",
            plugin="FFMPEG",
        )
    ):
        # convert numpy image to cv image
        # print(frame.shape)
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # display image and wait for key press

        print(type(frame))
        print(frame.shape)
        print(frame.dtype)
        # print(frame[0,0])

        model.add_image(frame)
        results = model.track()
        # print(results[0].shape, results[1].shape)
    

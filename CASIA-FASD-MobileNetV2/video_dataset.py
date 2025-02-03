import os
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, folder_path, num_frames=16, transform=None):
        self.folder_path = folder_path
        self.video_paths = []
        self.num_frames = num_frames  # Desired number of frames
        self.transform = transform

        # Get paths for all videos
        for identity in os.listdir(folder_path):
            identity_folder = os.path.join(folder_path, identity)
            for video in os.listdir(identity_folder):
                self.video_paths.append((os.path.join(identity_folder, video), identity))  # Store path with identity

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, identity = self.video_paths[idx]  # Unpack video path and identity
        frames = self.load_video(video_path)

        # Ensure exactly num_frames
        if frames.size(0) < self.num_frames:
            frames = torch.cat((frames, torch.zeros((self.num_frames - frames.size(0), frames.size(1), frames.size(2), frames.size(3)), dtype=frames.dtype)), dim=0)
        elif frames.size(0) > self.num_frames:
            frames = frames[:self.num_frames]

        if self.transform:
            frames = self.transform(frames)

        # Extract video filename (without extension) and assign labels based on specific filenames
        video_filename = os.path.basename(video_path).split('.')[0]

        # Custom label logic based on filenames
        label_1_videos = ["1", "2", "HR_1"]  # List of filenames that should have label 1
        label = 1 if video_filename in label_1_videos else 0

        return frames, label, identity  # Return frames, label, and identity


    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame extraction interval
        if total_frames < self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            interval = total_frames / self.num_frames  # Frame extraction interval
            frame_indices = [int(i * interval) for i in range(self.num_frames)]

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to the specific frame index
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (112, 112))  # Resize frame
                frame = torch.tensor(frame).permute(2, 0, 1)  # Change to (C, H, W)
                frames.append(frame)

        cap.release()

        # Stack frames to return as a tensor
        return torch.stack(frames) if frames else torch.empty(0)  # Return an empty tensor if no frames were captured

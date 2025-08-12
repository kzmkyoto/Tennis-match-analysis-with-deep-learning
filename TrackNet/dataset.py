import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import math
from PIL import Image


class TrackNetDataset(Dataset):
    def __init__(self, mode, path_dataset, input_height=360, input_width=640):
        self.path_dataset = path_dataset
        self.data = pd.read_csv(os.path.join(self.path_dataset, f"labels_{mode}"))
        self.HEIGHT = input_height
        self.WIDTH = input_width
        print(f"mode = {mode}, samples = {self.data.shape[0]}")

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        path_now, path_prev, path_prevprev, path_gt, x, y, status, visibility = self.data.loc[idx, :]

        path_now = os.path.join(self.path_dataset, path_now)
        path_prev = os.path.join(self.path_dataset, path_prev)
        path_prevprev = os.path.join(self.path_dataset, path_prevprev)
        path_gt = os.path.join(self.path_dataset, 'gts', path_gt)

        if math.isnan(x) or math.isnan(y):
            x = -1
            y = -1

        transform = transforms.Compose([
            transforms.Resize((self.HEIGHT, self.WIDTH)),
            transforms.ToTensor()          
        ])

        def pil_to_tensor_rgb(path):
            img = Image.open(path).convert("RGB")
            img = transform(img)
            return img

        gt_img = Image.open(path_gt).convert("RGB")
        gt_img = transform(gt_img)
        gt_img = gt_img[0, :, :]
        gt_img = gt_img.reshape(self.HEIGHT * self.WIDTH)
        gt_img = (gt_img * 255).byte()

        img_now = pil_to_tensor_rgb(path_now)
        img_prev = pil_to_tensor_rgb(path_prev)
        img_prevprev = pil_to_tensor_rgb(path_prevprev)

        imgs = torch.cat((img_now, img_prev, img_prevprev), dim=0)
        return imgs, gt_img, x, y, visibility

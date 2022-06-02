import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image

import os


class NYUDepthV2(Dataset):
    def __init__(self, root, rgb_transform=None,depth_transform=None):
        self.root = root
        self.rgb = os.path.join(self.root,"RGB")
        self.depth = os.path.join(self.root,"Depth")
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(os.listdir(self.rgb))

    def __getitem__(self, idx):
        # return a dictionary of tensors

        rgb_image = os.path.join(self.rgb,
                                "{}.jpg".format(idx))
        depth_image = os.path.join(self.depth,
                                  "{}.npy".format(idx))
        if self.rgb_transform :
            rgb_tensor = self.rgb_transform(read_image(rgb_image))
        else:
            rgb_tensor = read_image(rgb_image)
        
        if self.depth_transform :
            depth_tensor = self.depth_transform(torch.from_numpy(np.load(depth_image)).unsqueeze(0))
        else:
            depth_tensor = torch.from_numpy(np.load(depth_image)).unsqueeze(0)


        return {"rgb":rgb_tensor,"depth":depth_tensor}


# sample depth map with probability p 
def sample_depth_random(depth_full: torch.Tensor, p:float) -> torch.Tensor:
    mask = torch.rand_like(depth_full)
    zeros = torch.zeros_like(depth_full)
    #sampled = torch.where(mask<=p,depth_full,torch.tensor(0))
    sampled = torch.masked_fill(depth_full,mask > p, 0)

    return sampled


def mse(pred_depth: torch.Tensor,true_depth: torch.Tensor)-> float:
    res = pred_depth - true_depth
    
    return torch.sum(torch.abs(res)) # / num_pixels

def rmse(pred_depth: torch.Tensor,true_depth: torch.Tensor)-> float:
    res = pred_depth - true_depth
    
    return torch.sum(res * res) # / num_pixels

def abs_rel(pred_depth: torch.Tensor,true_depth: torch.Tensor)-> float:
    res = pred_depth - true_depth
    
    return torch.sum(torch.abs(res/true_depth)) # / num_pixels

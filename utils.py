import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

import os

class NYUDepthV2(Dataset):
    def __init__(self, root, shape=None,rgb_transform=None,depth_transform=None):
        self.root = root
        self.rgb = os.path.join(self.root,"RGB")
        self.depth = os.path.join(self.root,"Depth")
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        if shape is not None: self.resize = transforms.Resize(shape)
        else: self.resize = None

    def __len__(self):
        return len(os.listdir(self.rgb))

    def __getitem__(self, idx):
        # return a dictionary of tensors

        rgb_image_path = os.path.join(self.rgb,
                                "{}.jpg".format(idx))
        depth_image_path = os.path.join(self.depth,
                                  "{}.npy".format(idx))
        sample = {}

        rgb_tensor = read_image(rgb_image_path) / 255.
        depth_tensor = torch.from_numpy(np.load(depth_image_path)).unsqueeze(0) / 255.

        if self.resize:
            rgb_tensor = self.resize(rgb_tensor)
            depth_tensor = self.resize(depth_tensor)

        sample["rgb"] = rgb_tensor
        sample["depth"] = depth_tensor

        if self.rgb_transform :
            rgb_t = self.rgb_transform(rgb_tensor)
            sample["rgb_t"] = rgb_t
        
        if self.depth_transform :
            depth_t = self.depth_transform(depth_tensor)
            sample["depth_t"] = depth_t

        return sample


# sample depth map with probability p
# if p is an interval then the sampling will happen with a probability chosen in that interval
def sample_depth_random(depth_full: torch.Tensor, p) -> torch.Tensor:
    
    mask = torch.rand_like(depth_full)

    # if given an interval sample from the interval
    if type(p) == tuple: 
        thresh = torch.rand((1,1))*(p[1]-p[0]) + p[0]
    else:
        # if given single value value specified use that to create the mask
        thresh = p

    # print(thresh)

    sampled = torch.masked_fill(depth_full,mask > thresh, 0)

    return sampled


mse = torch.nn.MSELoss()

rmse = lambda predicted,true: torch.sqrt(mse(predicted,true))
lrmse = lambda predicted,true: torch.log10(rmse(predicted,true))

def d_accuracy(predicted,true,threshold=1.25,pow=1):

    # there are some negative values in the prediction (very close to 0)
    # this can cause problems when dividing
    predicted[predicted<0] = 0.0

    P = torch.numel(predicted)
    ratio = predicted / true
    max_ratio =  torch.where(ratio < 1,torch.pow(ratio,-1),ratio) # flip when ratio < 1

    return torch.sum(max_ratio<threshold**pow) / P # count the elements that satisfy the predicate


def count_params(model):
    sum(p.numel() for p in model.parameters())


def train(model,n_epochs,loss_fn,optimizer,loader,
         device=torch.device("cuda:0"),
         save_name=None,
         log=False): # TODO implement a log function

    # set model to train mode 
    # mainly affects batchnorm
    model.train()


    i = 0
    for j in range(n_epochs):

        for batch in loader:
            optimizer.zero_grad()

            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            depth_degraded = batch["depth_t"].to(device)

            out = model(rgb,depth_degraded)

            loss = loss_fn(depth,out)
            loss.backward()

            optimizer.step()
            i += 1

            if log and (i % 500 == 0): # log loss every 500 batches
                print("{}: loss: {}".format(i,loss))
        # save once every epoch
        if save_name is not None and (j % 15 == 0):
            torch.save(model.state_dict(),
                "weights/{}_{}.pth"\
                .format(save_name,j+1))

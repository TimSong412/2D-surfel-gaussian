#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchmetrics.functional.image import image_gradients
from .graphics_utils import fov2focal
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def norm_loss(P, M, depth, viewpoint_cam):
    if torch.norm(depth) < 0.00001:    
        print(depth)

        exit(9)
    fx = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width)
    fy = fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height)
    dZx, dZy = image_gradients(depth.unsqueeze(0))
    dZx = dZx.squeeze()
    dZy = dZy.squeeze()
    grad_x = torch.stack([depth.squeeze()/fx, torch.zeros_like(depth.squeeze()), dZx])
    grad_y = torch.stack([torch.zeros_like(depth.squeeze()), depth.squeeze()/fy, dZy])
    normal = torch.cross(grad_x, grad_y, dim=0) # normal in image space
    
    _, H, W = normal.size()
    normal = normal.view(3, -1).transpose(0, 1).unsqueeze(2)  # Now shape [HxW, 3, 1]

    # Perform batch matrix multiplication
    normal = torch.matmul(torch.Tensor(torch.inverse(viewpoint_cam.projection_matrix[:3,:3])).cuda(), normal)  # Matrix is [3, 3], tensor is [HxW, 3, 1]

    normal = normal.squeeze(2).transpose(0, 1).view(3, H, W) # normal in camera space

    normal = normal / torch.norm(normal, dim=0, keepdim=True) # normalize
    
    return (P + (M * normal).sum(dim=0)).mean()


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
import math

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
    
def norm_loss(P, M, depth, fx, fy, W, H):
    cx=W/2.0
    cy=H/2.0

    x=torch.arange(W).reshape(1, -1).repeat(H, 1).to(depth.device)
    y=torch.arange(H).reshape(-1, 1).repeat(1, W).to(depth.device)
    x=(x-cx)/fx
    y=(y-cy)/fy
    view_ray = torch.stack([x, y, torch.ones_like(x)]).to(depth.device)
    view_ray = view_ray / torch.norm(view_ray, dim=0, keepdim=True)
    
    xyz = torch.stack([x*depth.squeeze(), y*depth.squeeze(), depth.squeeze()])
    _, dPy, dPx = torch.gradient(xyz)
    normal = torch.cross(dPx, dPy, dim=0)
    normal = normal / torch.norm(normal, dim=0, keepdim=True)
    angle = torch.sum(normal * view_ray, dim=0)
    normal[:, angle > 0] *= -1.0

    
    return (P - (M * normal).sum(dim=0, keepdim=True)).mean(), normal, (P - (M * normal).sum(dim=0, keepdim=True))/2.0



def normal_loss_2DGS(P, M, depth, view):
    W = view.image_width
    H = view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    cx=W/2.0
    cy=H/2.0

    x=torch.arange(W).reshape(1, -1).repeat(H, 1).to(depth.device)+0.5
    y=torch.arange(H).reshape(-1, 1).repeat(1, W).to(depth.device)+0.5
    x=(x-cx)/fx
    y=(y-cy)/fy
    view_ray = torch.stack([x, y, torch.ones_like(x)]).to(depth.device)
    view_ray = view_ray / torch.norm(view_ray, dim=0, keepdim=True)
    
    xyz_ = torch.stack([x*depth.squeeze(), y*depth.squeeze(), depth.squeeze()])
    _, dPy, dPx = torch.gradient(xyz_)

    normal, xyz = depth_to_normal(view, depth)
    normal = normal.permute(2, 0, 1)
    xyz = xyz.permute(2, 0, 1)
    return (P - (M * normal).sum(dim=0, keepdim=True)).mean(), normal, (P - (M * normal).sum(dim=0, keepdim=True))/2.0



def depths_to_points(view, depthmap):
    # c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda()
    rays_d = points @ intrins.inverse().T
    # rays_o = torch.zeros_like(rays_d)
    points = depthmap.reshape(-1, 1) * rays_d #+ rays_o
    return points


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

def get_edge_map(gt_image):
    """
    return the exp(-grad) of the image
    edge will have small value
    """
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
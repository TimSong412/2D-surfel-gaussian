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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import open3d as o3d
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import fov2focal


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depth= render_pkg["depth"]
        radii= render_pkg["radii"]
        torch.cuda.synchronize()
        visualize(rendering,depth,radii,view,idx, depth_path)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

def visualize(rendering,depth,radii,view,idx, depth_path):
# def visualize(idx=0):
    '''
    Input:
    rendering: (3,H,W) tensor
    depth: (1,H,W) tensor
    K: (4,4) tensor
    '''
    
    scale=view.scale
    H,W=rendering.shape[1:]
    fx=fov2focal(view.FoVx, W)
    fy=fov2focal(view.FoVy, H)
    cx=W/2
    cy=H/2
    # torch.save(rendering, os.path.join(depth_path, 'rendering.pt'))
    # torch.save(depth, os.path.join(depth_path, 'depth.pt'))
    # K=torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # torch.save(K, os.path.join(depth_path, 'K.pt'))
    # torch.save(radii, os.path.join(depth_path, 'raddi.pt'))   
    # load data 
    # rendering = torch.load('test/rendering.pt')
    # depth = torch.load('test/depth.pt')
    # K = torch.load('test/K.pt')
    # radii = torch.load('test/raddi.pt')
    # fx=K[0,0]
    # fy=K[1,1]
    # cx=K[0,2]
    # cy=K[1,2]
    # print(rendering.shape)
    # W=rendering.shape[2]
    # H=rendering.shape[1]
    x=torch.arange(W).reshape(1, -1).repeat(H, 1)
    y=torch.arange(H).reshape(-1, 1).repeat(1, W)
    x=(x-cx)/fx
    y=(y-cy)/fy

    Z=depth.squeeze().cpu().numpy()
    X=(x*Z).numpy()
    Y=(y*Z).numpy()

    valid_mask = (Z>0)
    X=X[valid_mask]
    Y=Y[valid_mask]
    Z=Z[valid_mask]

    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(np.stack((X, Y, Z), axis=-1).reshape(-1, 3))
    colors = rendering.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors=np.clip(colors, 0.0, 1.0)

    colors = colors[valid_mask.flatten()]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(os.path.join(depth_path, f"output{idx:05d}.ply"), pcd)
    pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    # visualize()
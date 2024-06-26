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
from wis3d import Wis3D
from torchmetrics.functional.image import image_gradients
import matplotlib.pyplot as plt
from utils.loss_utils import norm_loss


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    v3d = Wis3D("dbg", model_path.strip("output/")[:10], "xyz")
    print("wis3d dir: ", model_path.strip("output/")[:10])
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    depthmap_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depthmap")
    normalmap_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normalmap")
    depthfile_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depthfile")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(depthmap_path, exist_ok=True)
    makedirs(normalmap_path, exist_ok=True)
    makedirs(depthfile_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # torch.cuda.empty_cache()
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depth= render_pkg["depth"]
        radii= render_pkg["radii"]
        ray_P = render_pkg["ray_P"]
        ray_M = render_pkg["ray_M"]
    

        if "normal" in render_pkg.keys():
            normal = render_pkg["normal"]
        else:
            normal = None
        
        visualize(rendering,depth,view,idx, depth_path, normal=normal, vis=v3d, normal_path=normalmap_path, ray_P=ray_P, ray_M=ray_M)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        depthfile = depth.cpu().numpy()
        np.savez_compressed(os.path.join(depthfile_path, '{0:05d}'.format(idx) + ".npz"), depth=depthfile)

        depth = torch.clamp(depth, 1, 15)
        normed_depth = (depth - depth.min()) / (depth.max() - depth.min())
        colored_depth = plt.cm.jet(normed_depth.squeeze().cpu().numpy())[..., :3]
        
        torchvision.utils.save_image(torch.from_numpy(colored_depth).permute(2, 0, 1), os.path.join(depthmap_path, '{0:05d}'.format(idx) + ".png"))

        # break
    
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

def visualize(rendering,depth,view,idx, depth_path, normal=None, vis: Wis3D =None, normal_path=None, ray_P=None, ray_M=None):
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
    print("fx: ", fx, "fy: ", fy)
    cx=W/2
    cy=H/2

    Ln, normal_L, lossmap = norm_loss(ray_P, ray_M, depth, fx, fy, W, H)
    print("Ln: ", Ln)
    

    x=torch.arange(W).reshape(1, -1).repeat(H, 1)
    y=torch.arange(H).reshape(-1, 1).repeat(1, W)
    x=(x-cx)/fx
    y=(y-cy)/fy

    Z=depth.squeeze().cpu().numpy()
    X=(x*Z).numpy()
    Y=(y*Z).numpy()

    xyz = torch.from_numpy(np.stack((X, Y, Z)))
    # dPx = torch.gradient(xyz, dim=2)[0]
    # dPy = torch.gradient(xyz, dim=1)[0]
    _, dPy, dPx = torch.gradient(xyz)
    dPx = dPx / torch.norm(dPx, dim=0, keepdim=True)
    dPy = dPy / torch.norm(dPy, dim=0, keepdim=True)
    normal_P = torch.cross(dPx, dPy, dim=0).cuda()
    normal_P = normal_P / torch.norm(normal_P, dim=0, keepdim=True)
    
    # dPx = np.gradient(xyz, axis=0)
    # dPy = np.gradient(xyz, axis=1)
    # dPx = dPx / np.linalg.norm(dPx, axis=-1, keepdims=True)
    # dPy = dPy / np.linalg.norm(dPy, axis=-1, keepdims=True)
    # normal_P = np.cross(dPx, dPy, axis=-1)
    # normal_P = normal_P / np.linalg.norm(normal_P, axis=-1, keepdims=True)

    # view_ray = np.stack([x.numpy(), y.numpy(), np.ones_like(x)], axis=-1)
    # view_ray = view_ray / np.linalg.norm(view_ray, axis=-1, keepdims=True)
    # angle = np.sum(normal_P * view_ray, axis=-1)
    # normal_P[angle>0, :] *= -1.0

    # torchvision.utils.save_image(torch.from_numpy((1-normal_P)/2).permute(2, 0, 1), os.path.join(normal_path, f"normal_P{idx:05d}.png"))


    valid_mask = (Z>0)
    X=X[valid_mask]
    Y=Y[valid_mask]
    Z=Z[valid_mask]

    # depth_D = depth.clone().detach().unsqueeze(0)
    # dZx, dZy = image_gradients(depth_D)
    # dZx = dZx.squeeze()
    # dZy = dZy.squeeze()
    # grad_x = torch.stack([depth_D.squeeze()/fx, torch.zeros_like(depth_D.squeeze()), dZx])
    # grad_y = torch.stack([torch.zeros_like(depth_D.squeeze()), depth_D.squeeze()/fy, dZy])
    # grad_x = grad_x / torch.norm(grad_x, dim=0, keepdim=True)
    # grad_y = grad_y / torch.norm(grad_y, dim=0, keepdim=True)
    # normal_D = torch.cross(grad_x, grad_y, dim=0)
    # normal_D = normal_D / torch.norm(normal_D, dim=0, keepdim=True)

    # compute the angle between the depth normal and view ray, flip the normal if the angle is less than 90 degrees
    view_ray = torch.stack([x, y, torch.ones_like(x)]).cuda()
    view_ray = view_ray / torch.norm(view_ray, dim=0, keepdim=True)
    # angle = torch.sum(normal_D * view_ray, dim=0)
    # normal_D[:, angle>0] *= -1.0

    angle = torch.sum(normal_P * view_ray, dim=0)
    normal_P[:, angle>0] *= -1.0
    normal_RGB = (1-normal_P) / 2.0
    torchvision.utils.save_image(normal_RGB, os.path.join(normal_path, f"normal_P{idx:05d}.png"))

    print("norm err: ", torch.norm(normal_P - normal_L))
    
    

    # normal_RGB = (1-normal_D) / 2.0
    # torchvision.utils.save_image(normal_RGB, os.path.join(normal_path, f"N{idx:05d}.png"))

    Z = np.clip(Z, 1.0, 15)
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(np.stack((X, Y, Z), axis=-1).reshape(-1, 3))
    colors = rendering.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors=np.clip(colors, 0.0, 1.0)

    colors = colors[valid_mask.flatten()]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(os.path.join(depth_path, f"output{idx:05d}.ply"), pcd)

    if normal is not None and vis is not None:
        color_normal = (1-normal) / 2.0
        torchvision.utils.save_image(color_normal, os.path.join(normal_path, f"rendernormal_{idx:05d}.png"))
        # vis.set_scene_id(idx)
        normal = normal.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()[valid_mask.flatten()]
        vis.add_point_cloud(np.stack((X, Y, Z), axis=-1).reshape(-1, 3), colors= colors, name="pointcloud")
        stride = 111
        vis.add_lines(np.stack((X, Y, Z), axis=-1).reshape(-1, 3)[::stride], (np.stack((X, Y, Z), axis=-1).reshape(-1, 3) + 0.2*normal.reshape(-1, 3))[::stride], name="normals")
        # normal_D = normal_D.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()[valid_mask.flatten()]
        # vis.add_lines(np.stack((X, Y, Z), axis=-1).reshape(-1, 3)[::100], (np.stack((X, Y, Z), axis=-1).reshape(-1, 3) + 0.2*normal_D.reshape(-1, 3))[::100], name="normals_D")
        # grad_x = grad_x.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()[valid_mask.flatten()]
        # vis.add_lines(np.stack((X, Y, Z), axis=-1).reshape(-1, 3)[::100], (np.stack((X, Y, Z), axis=-1).reshape(-1, 3) + 0.2*grad_x.reshape(-1, 3))[::100], name="grad_x")
        # grad_y = grad_y.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()[valid_mask.flatten()]
        # vis.add_lines(np.stack((X, Y, Z), axis=-1).reshape(-1, 3)[::100], (np.stack((X, Y, Z), axis=-1).reshape(-1, 3) + 0.2*grad_y.reshape(-1, 3))[::100], name="grad_y")
        # dPx = dPx.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()[valid_mask.flatten()]
        # vis.add_lines(np.stack((X, Y, Z), axis=-1).reshape(-1, 3)[::stride], (np.stack((X, Y, Z), axis=-1).reshape(-1, 3) + 0.2*dPx.reshape(-1, 3))[::stride], name="dPx")
        # dPy = dPy.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()[valid_mask.flatten()]
        # vis.add_lines(np.stack((X, Y, Z), axis=-1).reshape(-1, 3)[::stride], (np.stack((X, Y, Z), axis=-1).reshape(-1, 3) + 0.2*dPy.reshape(-1, 3))[::stride], name="dPy")
        normal_P = normal_P.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()[valid_mask.flatten()]
        vis.add_lines(np.stack((X, Y, Z), axis=-1).reshape(-1, 3)[::stride], (np.stack((X, Y, Z), axis=-1).reshape(-1, 3) + 0.2*normal_P.reshape(-1, 3))[::stride], name="normal_P")




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
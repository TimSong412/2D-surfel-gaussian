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
from utils.graphics_utils import fov2focal,getWorld2View

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    colors = []
    depths = []
    poses = []
    K = torch.zeros(4, 4)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        colors.append(rendering)
        depths.append(render_pkg["depth"])
        pose=getWorld2View(view.R, view.T)
        poses.append(pose) 
        K = get_K(render_pkg, view, idx)
        # K = visualize_pc(render_pkg, view, idx)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path,
                                    '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    name="2"
    torch.save(depths, f'test_new/{name}/data/depths.pt')
    torch.save(colors, f'test_new/{name}/data/colors.pt')
    torch.save(poses, f'test_new/{name}/data/poses.pt')
    torch.save(K, f'test_new/{name}/data/K.pt')
    mesh_extraction(depths, colors, K, poses)

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

def get_K(render_pkg,view,idx):
    '''
    Input:
    rendering_pkg: dict
        rendering_pkg["render"]: torch.Tensor, shape (3, H, W)
        rendering_pkg["depth"]: torch.Tensor, shape (1, H, W)
    view: Camera object
    idx: int

    return: torch.Tensor, shape (4, 4)
    '''
    rendering = render_pkg["render"]
    depth= render_pkg["depth"]
    scale=view.scale
    H,W=rendering.shape[1:]
    # camera intrinsic
    fx=fov2focal(view.FoVx, W)
    fy=fov2focal(view.FoVy, H)
    cx=W/2
    cy=H/2
    K=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def visualize_pc(render_pkg,view,idx):
    # def visualize(idx=0):
    '''
    Input:
    rendering_pkg: dict
        rendering_pkg["render"]: torch.Tensor, shape (3, H, W)
        rendering_pkg["depth"]: torch.Tensor, shape (1, H, W)
    view: Camera object
    idx: int

    return: torch.Tensor, shape (4, 4)
    '''
    rendering = render_pkg["render"]
    depth= render_pkg["depth"]
    scale=view.scale
    H,W=rendering.shape[1:]
    # camera intrinsic
    fx=fov2focal(view.FoVx, W)
    fy=fov2focal(view.FoVy, H)
    cx=W/2
    cy=H/2
    K=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # torch.save(rendering, 'test/rendering.pt')
    # torch.save(depth, 'test/depth.pt')
    # torch.save(K, 'test/K.pt')
    # load data
    # rendering = torch.load('test/rendering.pt')
    # depth = torch.load('test/depth.pt')
    # K = torch.load('test/K.pt')
    # fx=K[0,0]
    # fy=K[1,1]
    # cx=K[0,2]
    # cy=K[1,2]
    # W=rendering.shape[2]
    # H=rendering.shape[1]
    x=torch.arange(W).reshape(1, -1).repeat(H, 1)
    y=torch.arange(H).reshape(-1, 1).repeat(1, W)
    X=(x-cx)/fx
    Y=(y-cy)/fy

    Z=depth.squeeze().cpu().numpy()
    X=(X*Z).numpy()
    Y=(Y*Z).numpy()
    # create point cloud
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(np.stack((X, Y, Z), axis=-1).reshape(-1, 3))
    colors = rendering.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors_clip=np.clip(colors, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors_clip)
    o3d.io.write_point_cloud(f"test/point_cloud_result/3-Ld/pc_{idx}.ply", pcd)
    return K

def mesh_extraction(depths, colors, K, RTs):
    """
    Creates a mesh from a series of depth images.
    
    Parameters:
    - depths: List of depth images. (n, 1, H, W)
    - colors: List of color images. (n, 3, H, W)
    - K: Camera intrinsic matrix. (3, 3)
    - RTs: Camera poses corresponding to each depth image.  (n, 4, 4)

    """
    # Voxel size used in TSDF fusion.
    voxel_length = 0.006
    # Truncation threshold for TSDF.
    sdf_trunc = 0.05
    depth_trunc = 5.5
    # Initialize a TSDF volume
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    H,W = depths[0].shape[1:]
    # Process each depth image
    for i in range(len(depths)):
        depth = depths[i]
        RT = RTs[i]
        color = colors[i]
        print("Integrate {:d}-th image into the volume.".format(i))
        # Create an Open3D depth image
        color_np=color.reshape(H,W,3).cpu().numpy()
        color_clip=np.clip(color_np, 0.0, 1.0)*255
        depth_o3d = o3d.geometry.Image(
            depth.squeeze(0).cpu().numpy().astype(np.float32))
        color_o3d = o3d.geometry.Image(color_clip.astype(np.uint8))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False)
        # Create a camera intrinsic object
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(W,H,fx=K[0, 0],fy=K[1, 1],cx=K[0, 2],cy=K[1, 2])
        # Integrate the current depth image into the TSDF volume
        RT_o3d=RT.astype(np.float64)
        tsdf_volume.integrate(rgbd_image, intrinsic_o3d,RT_o3d)

    # Extract the mesh from the TSDF volume
    mesh = tsdf_volume.extract_triangle_mesh()
    # Compute vertex normals to improve rendering
    mesh.compute_vertex_normals()
    # mesh_simplfied = mesh.simplify_quadric_decimation(target_number_of_triangles =65345254//10)
    # print(f'Simplified mesh has {len(mesh_simplfied.vertices)} vertices and {len(mesh_simplfied.triangles)} triangles')
    # mesh_simplfied = mesh.simplify_quadric_decimation(target_number_of_triangles =65345254//2)
    # print(f'Simplified mesh has {len(mesh_simplfied.vertices)} vertices and {len(mesh_simplfied.triangles)} triangles')
    print(f'Original mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} trianglpes')

    mesh_simplfied = mesh.simplify_quadric_decimation(target_number_of_triangles =42715716//20)
    # print(f'Simplified mesh has {len(mesh_simplfied.vertices)} vertices and {len(mesh_simplfied.triangles)} triangles')
    # Save the mesh to a file
    name="2"
    o3d.io.write_triangle_mesh(f"test_new/{name}/mesh_result/mesh_voxel_{voxel_length}_sdf_trunc_{sdf_trunc}_dpepth_trunc_{depth_trunc}.ply", mesh_simplfied, write_ascii=False, write_vertex_colors=False)
    pass


def simplify_mesh(file_name, mesh, n):
    """
    Simplifies a mesh using quadric edge collapse decimation.
    
    Parameters:
    - mesh: The input mesh.
    - target_reduction: The target reduction factor.
    
    Returns:
    - The simplified mesh.
    """
    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=n)
    o3d.io.write_triangle_mesh(f"{file_name}simplified", simplified_mesh, write_ascii=False, write_vertex_colors=False)

    return simplify_mesh

if __name__ == "__main__":
    # file_name="test_new/mesh_result/2-without-Ld/mesh_voxel_0.004_sdf_trunc_0.02_dpepth_trunc_6.ply"
    # mesh = o3d.io.read_triangle_mesh(file_name)
    # mesh=simplify_mesh(file_name,mesh,n=65345254//5)
    load=True
    if load:
        # visualize()
        name="2"
        depths=torch.load(f'test_new/{name}/data/depths.pt')
        poses=torch.load(f'test_new/{name}/data/poses.pt')
        colors=torch.load(f'test_new/{name}/data/colors.pt')
        K=torch.load(f'test_new/{name}/data/K.pt')
        mesh_extraction(depths, colors,K, poses)
    else:
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

import numpy as np
from pathlib import Path
from colmap_io import Camera, BaseImage, CameraModel, Point3D, rotmat2qvec, write_cameras_text, write_images_text, write_points3D_text

from wis3d import Wis3D
import OpenEXR
import Imath
import trimesh
import imageio.v2 as imageio
import json


def read_exr(s, width, height):
    mat = np.fromstring(s, dtype=np.float32)
    mat = mat.reshape(height, width)
    return mat



def exr_to_tensor(exr_path, single=True):
    exr_image = OpenEXR.InputFile(exr_path)
    dw = exr_image.header()['dataWindow']
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)


    b, g, r = [read_exr(s, width, height) for s in exr_image.channels('BGR', Imath.PixelType(Imath.PixelType.FLOAT))]
    # dmap = np.asarray(dmap,np.float64)

    if single:
        return b
    
    return np.stack([r, g, b], axis=0)


cam_model = CameraModel(model_id=1, model_name="PINHOLE", num_params=4),

def create_cams_imgs(datadir:Path, targetdir:Path=None):
    transforms = json.load((datadir / "transforms.json").open())
    img = ["img27.jpg", "img40.jpg", "img53.jpg", "img8.jpg", "img37.jpg", "img12.jpg"]
    cameras = {}
    images = {}
    viewid = 0
    outdir = targetdir / "sparse" / "0"
    outdir.mkdir(parents=True, exist_ok=True)
    for vid, viewname in enumerate(img):
        imgname = f"{vid:04d}.jpg"
        
        
        for frame in transforms['frames']:
            if frame['file_path'] == viewname:
                break
        depth_file = datadir / frame['depth_file_path']

        imgfile = datadir / viewname
        img = imageio.imread(imgfile.__str__())
        imageio.imwrite((outdir / imgname).__str__(), img)
        
        
        # intrin: [fx, fy, cx, cy]
        intrin = [transforms['fl_x'], transforms['fl_y'], transforms['cx'], transforms['cy']]
        cam_pose = np.eye(4)
        cam_pose[0] = np.array(frame['transform_matrix'][0])
        cam_pose[1] = np.array(frame['transform_matrix'][1])
        cam_pose[2] = np.array(frame['transform_matrix'][2])
        cam_pose[3] = np.array(frame['transform_matrix'][3])

        # convert x-right, y-up, z-backs to x-right, y-down, z-forward
        cam_pose[0:3, 1:3] *= -1

        R = cam_pose[:3, :3]
        t = cam_pose[:3, 3]
       
        view_cam = Camera(viewid, "PINHOLE", width=int(transforms['w']), height=int(transforms['h']), params=intrin)
        cameras[viewid] = view_cam
        view_img = BaseImage(viewid, qvec=rotmat2qvec(R), tvec=t, camera_id=viewid, name=imgfile.__str__(), xys=[[0, 0]], point3D_ids=[0])
        images[viewid] = view_img
        viewid += 1
    write_images_text(images, outdir / "images.txt")
    write_cameras_text(cameras, outdir / "cameras.txt")

def read_poses_depth(datadir:Path, w3d:Wis3D=None):
    img = ["img27.jpg", "img40.jpg", "img53.jpg", "img8.jpg", "img37.jpg", "img12.jpg"]
    transforms = json.load((datadir / "transforms.json").open())
    poses = []
    depths = []
    intrins = []
    rgbs = []
    cid = 0
    for vid, viewname in enumerate(img):
        for frame in transforms['frames']:
            if frame['file_path'] == viewname:
                break
        depth_file = datadir / frame['depth_file_path']
        cam_pose = np.eye(4)
        cam_pose[0] = np.array(frame['transform_matrix'][0])
        cam_pose[1] = np.array(frame['transform_matrix'][1])
        cam_pose[2] = np.array(frame['transform_matrix'][2])
        cam_pose[3] = np.array(frame['transform_matrix'][3])

        # convert x-right, y-up, z-backs to x-right, y-down, z-forward
        cam_pose[0:3, 1:3] *= -1
        # cam_pose = cam_pose[np.array([0, 2, 1, 3]), :]

        # convert world to camera
        # R = cam_pose[:3, :3]
        # t = cam_pose[:3, 3]
        # cam_pose[:3, 3] = -R.T @ t
        # cam_pose[:3, :3] = R.T
        

        # intrin: [fx, fy, cx, cy]
        intrin = [transforms['fl_x'], transforms['fl_y'], transforms['cx'], transforms['cy']]


        if w3d is not None:
            w3d.add_lines(start_points=[cam_pose[:3, 3]], end_points=[cam_pose[:3, 3] + cam_pose[:3, 0]], name=f"camx{cid:03d}")
            w3d.add_lines(start_points=[cam_pose[:3, 3]], end_points=[cam_pose[:3, 3] + cam_pose[:3, 1]], name=f"camy{cid:03d}")
            w3d.add_lines(start_points=[cam_pose[:3, 3]], end_points=[cam_pose[:3, 3] + cam_pose[:3, 2]], name=f"camz{cid:03d}")
        
        poses.append(cam_pose)
        intrins.append(intrin)

        depth = np.load(depth_file).squeeze() 
        depths.append(depth)

        rgbfile = datadir / viewname
        image = imageio.imread(rgbfile.__str__())
        rgbs.append(image)
        cid += 1
        
    return poses, intrins, depths, rgbs
    

def create_pcd(datadir:Path):
    mesh = trimesh.load_mesh(datadir / "model.stl")
    points = mesh.vertices
    colors = mesh.visual.vertex_colors
    if len(points) > 1000:
        # randomly choose 3000 points
        idx = np.random.choice(len(points), 500, replace=False)
        points = points[idx]
        colors = colors[idx]

    pcd = {}
    for i, (point, color) in enumerate(zip(points, colors)):
        pcd[i] = Point3D(id=i, xyz=point, rgb=color[:3], error=0, image_ids=[0], point2D_idxs=[0])
    outdir = datadir / "sparse" / "0"
    outdir.mkdir(parents=True, exist_ok=True)
    write_points3D_text(pcd, outdir / "points3D.txt")

def fusepcd(camposes, camintrins, depth, rgbs):
    xyz = []
    colors = []
    for campose, camintrin, dmap, rgb in zip(camposes, camintrins, depth, rgbs):
        uvmap = np.mgrid[0:dmap.shape[0], 0:dmap.shape[1]][::-1].transpose(1, 2, 0).astype(np.float32)
        uvmap[..., 0] -= camintrin[2]
        uvmap[..., 1] -= camintrin[3]
        uvmap[..., 0] /= camintrin[0]
        uvmap[..., 1] /= camintrin[1]
        xyzmap = np.concatenate([uvmap, np.ones_like(uvmap[..., 0:1])], axis=-1) * (dmap[..., np.newaxis])
        
        valid_pts = dmap > 0
        xyzlist = xyzmap[valid_pts]
        rgblist = rgb[valid_pts]
        xyz_world = xyzlist @ campose[:3, :3].T + campose[:3, 3]
        xyz.append(xyz_world)
        colors.append(rgblist)

    # xyz = np.concatenate(xyz, axis=0)
    # colors = np.concatenate(colors, axis=0)
    return xyz, colors

def sample_pcd(pcd, colors, datadir:Path, targetdir:Path=None):
    outpcd = {}
    if len(pcd) > 1000:
        # randomly choose 3000 points
        idx = np.random.choice(len(pcd), 1000, replace=False)
        pcd = pcd[idx]
        colors = colors[idx]
    for i, (point, color) in enumerate(zip(pcd, colors)):
        outpcd[i] = Point3D(id=i, xyz=point, rgb=color[:3], error=0, image_ids=[0], point2D_idxs=[0])
    outdir = targetdir / "sparse" / "0"
    outdir.mkdir(parents=True, exist_ok=True)
    write_points3D_text(outpcd, outdir / "points3D.txt")


def create_colmap_for(datadir:Path):
    target_dir =Path("dataset/mugs")
    create_cams_imgs(datadir, targetdir=target_dir)
    v3d = Wis3D("visdir", "fuse")

    poses, intrins, depths, rgbs = read_poses_depth(datadir, w3d=v3d)

    pcd, color = fusepcd(poses, intrins, depths, rgbs)

    for i, (xyz, rgb) in enumerate(zip(pcd, color)):
        v3d.add_point_cloud(vertices=xyz[::10], colors=rgb[::10], name=f"pcd{i:03d}")

    
    pcd = np.concatenate(pcd, axis=0)
    color = np.concatenate(color, axis=0)

    
    

    
    sample_pcd(pcd, color, datadir, targetdir=target_dir)



if __name__ == "__main__":
    # datadir = Path("render/dataset/334e159972fd425d93f29d3c19c7f811")
    # create_cams_imgs(datadir)
    # # create_pcd(datadir)
    #     # read

    # v3d = Wis3D("visdir", "fuse")
    # poses, intrins, depths, rgbs = read_poses_depth(datadir, v3d)

    # pcd, color = fusepcd(poses, intrins, depths, rgbs)
    
    # mesh = trimesh.load_mesh(datadir / "model.stl")
    # points = mesh.vertices

    # v3d.add_point_cloud(vertices=points, name="model")
    # did=0
    # for xyz, rgb in zip(pcd, color):
    #     v3d.add_point_cloud(vertices=xyz, colors=rgb, name=f"pcd{did:03d}")
    #     did += 1
    
    # pcd = np.concatenate(pcd, axis=0)
    # color = np.concatenate(color, axis=0)
    
    # sample_pcd(pcd, color, datadir)
    # # data_dirs = list(Path("render/dataset").glob("*"))
    # data_dirs = [Path("render/dataset/334e159972fd425d93f29d3c19c7f811")]
    # for datadir in data_dirs:
    create_colmap_for(Path("dataset/colormugs"))
    # if not (datadir / "index.npy").exists():
        #     continue
        # create_cams_imgs(datadir)
        # poses, intrins, depths, rgbs = read_poses_depth(datadir)

        # pcd, color = fusepcd(poses, intrins, depths, rgbs)
        
        # mesh = trimesh.load_mesh(datadir / "model.stl")
        # pcd = np.concatenate(pcd, axis=0)
        # color = np.concatenate(color, axis=0)
        
        # sample_pcd(pcd, color, datadir)
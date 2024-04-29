import trimesh
import numpy as np
from wis3d import Wis3D
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
import json
from matplotlib import pyplot as plt

def quat2mat(quat: np.ndarray):
    # convert quaternion to rotation matrix
    w, x, y, z = quat
    return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])

def mat2quat(mat: np.ndarray):
    # convert rotation matrix to quaternion
    
    if 1 + mat[0, 0] + mat[1, 1] + mat[2, 2] < 1e-6:
        # w = 0
        if 1+mat[0, 0] - mat[1, 1] - mat[2, 2] < 1e-6:
            # x = 0
            y = np.sqrt(1-mat[0, 0] + mat[1, 1] - mat[2, 2]) / 2
            x = (mat[0, 1] + mat[1, 0]) / (4*y)
            z = (mat[1, 2] + mat[2, 1]) / (4*y)
            w = (mat[2, 0] - mat[0, 2]) / (4*y)
        else:
            x = np.sqrt(1+mat[0, 0] - mat[1, 1] - mat[2, 2]) / 2
            y = (mat[0, 1] + mat[1, 0]) / (4*x)
            z = (mat[0, 2] + mat[2, 0]) / (4*x)
            w = (mat[1, 2] - mat[2, 1]) / (4*x)
    else:
        w = np.sqrt(1 + mat[0, 0] + mat[1, 1] + mat[2, 2]) / 2
        x = (mat[2, 1] - mat[1, 2]) / (4*w)
        y = (mat[0, 2] - mat[2, 0]) / (4*w)
        z = (mat[1, 0] - mat[0, 1]) / (4*w)
    return np.array([w, x, y, z])

def normal2quat(normals):
    # the last column of the rotation matrix is the normal, other two axis are arbitrary
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    norm1 = []
    norm2 = []
    all_quat = []
    max_err = -100
    for normal in normals:
        # randomly find a vector that is vertical to the normal
        v1 = np.random.rand(3)*2 - 1.0
        v1 = v1 - np.dot(v1, normal) * normal
        v1 = v1 / np.linalg.norm(v1)
        norm1.append(v1)
        v2 = np.cross(normal, v1)
        norm2.append(v2)
        # create rotation matrix
        rot_mat = np.array([v1, v2, normal]).T  
        
        # convert to quaternion
        # rot = R.from_matrix(rot_mat)
        # quat = rot.as_quat()
        quat = mat2quat(rot_mat)
        
        all_quat.append(quat)
        mat = quat2mat(quat)
        err = np.sum(np.abs(rot_mat - mat))
        # print("Error: ", err)
        if err > max_err:
            max_err = err
    print("Max Error: ", max_err)
    return np.array(all_quat), np.array(norm1), np.array(norm2)



def gen_cube():
    # generate a cube gaussian spalting ply file
    mesh = trimesh.creation.box((1, 1, 1))
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    # save ply file
    mesh.export("cube.ply")
    print("vetices: ", len(mesh.vertices))

    # sample part of the vertex
    
    non_edge_mask = ((abs(mesh.vertices[:, 0]) + abs(mesh.vertices[:, 1]))!= 1) & (abs(mesh.vertices[:, 0])+abs(mesh.vertices[:, 2])!= 1) & (abs(mesh.vertices[:, 1])+abs(mesh.vertices[:, 2])!= 1)

    select_id = np.random.choice(sum(non_edge_mask), 10, replace=False)
    # select_id = [i for i in range(0, 1000, 5)]

    select_vert = mesh.vertices[non_edge_mask][select_id]
    select_norm = mesh.vertex_normals[non_edge_mask][select_id]
    # add to wis3d
    print("Add to Wis3D")
    v3d = Wis3D("dbg", "vis_normal", "xyz")
    v3d.add_mesh(mesh, "cube")
    # add vertex normal
    short_norm = select_norm * 0.01
    v3d.add_lines(np.array(select_vert), np.array(select_vert + short_norm), name=f"normals")

    # create gaussian splatting ply
    print("Create Gaussian Splatting Ply")
    list_attributes = ["x", "y", "z", "nx", "ny", "nz"]
    list_attributes += [f"f_dc_{i}" for i in range(3)]
    # list_attributes += [f"f_rest_{i}" for i in range(45)]
    list_attributes += ["opacity"]
    list_attributes += [f"scale_{i}" for i in range(3)]
    list_attributes += [f"rot_{i}" for i in range(4)]

    dtype_full = [(attr, 'f4') for attr in list_attributes]

    z_face = select_vert[:, 2] == select_vert[:, 2].min()
    y_face = select_vert[:, 1] == select_vert[:, 1].min()
    x_face = select_vert[:, 0] == select_vert[:, 0].min()
    zz_face = select_vert[:, 2] == select_vert[:, 2].max()
    yy_face = select_vert[:, 1] == select_vert[:, 1].max()
    xx_face = select_vert[:, 0] == select_vert[:, 0].max()

    xyz = np.array(select_vert) * 30
    normals = np.array(select_norm)
    # generate color in N * [0, 1]
    # f_dc = np.random.uniform(0, 1, (len(xyz), 3))
    f_dc = np.zeros((len(xyz), 3))

    f_dc[z_face] = [1, 0, 0]
    f_dc[y_face] = [0, 1, 0]
    f_dc[x_face] = [0, 0, 1]
    f_dc[zz_face] = [1, 1, 0]
    f_dc[yy_face] = [1, 0, 1]
    f_dc[xx_face] = [0, 1, 1]

    # generate grid texture for the cube
    freq = 10.0
    grid_length = (select_vert[:, 0].max() - select_vert[:, 0].min()) / freq
    vertmax = abs(select_vert[:, 0]).max()
    for idx, vert in enumerate(select_vert):
        x = vert[0]
        y = vert[1]
        z = vert[2]
        if abs(x) == vertmax:
            abs_z = z - select_vert[:, 2].min()
            pos_z = abs_z // grid_length
            abs_y = y - select_vert[:, 1].min()
            pos_y = abs_y // grid_length
            if (pos_z % 2 + pos_y % 2) % 2 == 0:
                f_dc[idx] = [1, 1, 1]
        if abs(y) == vertmax:
            abs_z = z - select_vert[:, 2].min()
            pos_z = abs_z // grid_length
            abs_x = x - select_vert[:, 0].min()
            pos_x = abs_x // grid_length
            if (pos_z % 2 + pos_x % 2) % 2 == 0:
                f_dc[idx] = [1, 1, 1]
        if abs(z) == vertmax:
            abs_y = y - select_vert[:, 1].min()
            pos_y = abs_y // grid_length
            abs_x = x - select_vert[:, 0].min()
            pos_x = abs_x // grid_length
            if (pos_y % 2 + pos_x % 2) % 2 == 0:
                f_dc[idx] = [1, 1, 1]

    f_rest = np.zeros((len(xyz), 45))
    opacities = np.ones((len(xyz), 1)) * 2
    scale = np.random.uniform(0, 1, (len(xyz), 3))
    scale[:, 2] = 0
    # generate quaternion based on normal
    rotation, norm1, norm2 = normal2quat(normals)

    # add new axies to vis3d
    norm1 *= scale[:, 0:1]
    norm2 *= scale[:, 1:2]
    v3d.add_lines(select_vert, select_vert + norm1, name=f"norm1")
    v3d.add_lines(select_vert, select_vert + norm2, name=f"norm2")

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write("cube_gaussian.ply")


def gen_capsule():
    # generate a cube gaussian spalting ply file
    mesh = trimesh.creation.capsule(0.2, 0.5)
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()
    # save ply file
    mesh.export("capsule.ply")
    print("vetices: ", len(mesh.vertices))

    # sample part of the vertex
    
    select_id = np.random.choice(len(mesh.vertices), 100000)
    # select_id = [i for i in range(0, 1000, 5)]

    select_vert = mesh.vertices[select_id]
    select_norm = mesh.vertex_normals[select_id]
    # add to wis3d
    print("Add to Wis3D")
    v3d = Wis3D("dbg", "vis_normal_capsule", "xyz")
    v3d.add_mesh(mesh, "capsule")
    # add vertex normal
    
    v3d.add_lines(np.array(select_vert), np.array(select_vert + select_norm*0.1), name=f"normals")

    # create gaussian splatting ply
    print("Create Gaussian Splatting Ply")
    list_attributes = ["x", "y", "z", "nx", "ny", "nz"]
    list_attributes += [f"f_dc_{i}" for i in range(3)]
    # list_attributes += [f"f_rest_{i}" for i in range(45)]
    list_attributes += ["opacity"]
    list_attributes += [f"scale_{i}" for i in range(3)]
    list_attributes += [f"rot_{i}" for i in range(4)]

    dtype_full = [(attr, 'f4') for attr in list_attributes]

    xyz = np.array(select_vert) * 20
    normals = np.array(select_norm)
    # generate color in N * [0, 1]
    # f_dc = np.random.uniform(0, 1, (len(xyz), 3))
    f_dc = np.zeros((len(xyz), 3))

    max_z = xyz[:, 2].max()
    min_z = xyz[:, 2].min()
    for i, z in enumerate(xyz[:, 2]):
        relative_z = (z - min_z) / (max_z - min_z)
        color = plt.cm.hsv(relative_z)[:3]
        f_dc[i] = color

    f_rest = np.zeros((len(xyz), 45))
    opacities = np.ones((len(xyz), 1))*10
    scale = np.random.uniform(-3, -2, (len(xyz), 3))
    scale[:, 2] = -3
    # generate quaternion based on normal
    rotation, norm1, norm2 = normal2quat(normals)

    # add new axies to vis3d
    norm1 *= scale[:, 0:1]
    norm2 *= scale[:, 1:2]
    v3d.add_lines(select_vert, select_vert + norm1, name=f"norm1")
    v3d.add_lines(select_vert, select_vert + norm2, name=f"norm2")

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write("capsule_gaussian.ply")


def resize_camera():
    with open("output/cube/cameras.json", "r") as f:
        cameras = json.load(f)  
    for cam in cameras:
        cam["position"] = [i*10 for i in cam["position"]]
    with open("output/cube/cameras_new.json", "w") as f:
        json.dump(cameras, f)

    

if __name__ == "__main__":
    # gen_cube()
    gen_capsule()
    print("Done!")
    # resize_camera()
import trimesh
import numpy as np
from PIL import Image
from trimesh.visual.texture import SimpleMaterial, TextureVisuals
import pyrender
import numpy as np

def create_face(vertices, face_indices, image_path, uvs):
    img = Image.open(image_path)
    material = SimpleMaterial(image=img)
    texture_visuals = TextureVisuals(uv=uvs, image=img, material=material)

    face_mesh = trimesh.Trimesh(vertices=vertices[face_indices], 
                                faces=np.array([[0, 1, 2], [2, 3, 0]]),
                                visual=texture_visuals,
                                process=False)
    return face_mesh

vertices = np.array([
    [-1, -1, -1], [-1, -1,  1], [-1,  1, 1], [-1,  1,  -1],  # 后
    [ 1, -1, -1], [ 1, 1,  -1], [ 1,  1, 1], [ 1,  -1,  1],  # 前
    [-1, -1, 1], [1, -1,  1], [ 1, 1, 1], [ -1, 1,  1],  # 顶
    [-1,  -1, -1], [-1,  1,  -1], [ 1,  1, -1], [ 1,  -1, -1],  # 顶
    [1, -1, 1], [-1,  -1, 1], [ -1, -1, -1], [ 1,  -1, -1],  # 左
    [1, 1,  -1], [-1,  1, -1], [ -1, 1,  1], [ 1,  1,  1]   # 右
])


face_indices = [
    [0, 1, 2, 3],  # 后
    [4, 5, 6, 7],  # 前
    [8, 9, 10, 11], # 底
    [12, 13, 14, 15], # 顶
    [16, 17, 18, 19], # 左
    [20, 21, 22, 23]  # 右
]

uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

path='../images/'
image_paths = [path+'3.jpg', path+'4.jpg', path+'5.jpg', path+'6.png', path+'7.jpg', path+'8.jpg']

cube_faces = [create_face(vertices, face_indices[i], image_paths[i], uv) for i in range(6)]
combined_mesh = trimesh.util.concatenate(cube_faces)
combined_mesh.fix_normals()
combined_mesh.show()

combined_mesh.export('../model/textured_cube.obj', include_texture=True,include_normals=True)
combined_mesh.export('../model/textured_cube.ply')




import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt

def plot_all_poses(poses):
    num_poses = poses.shape[0]
    poses = np.array(poses, dtype=np.float32)

    translations = poses[:, :3, 3]

    origins = translations
    directions = np.einsum('ijk,k->ij', poses[:, :3, :3], np.array([0.0, 0.0, 1.0]))
    
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
                  directions[:, 0], directions[:, 1], directions[:, 2], length=0.4, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

mesh = trimesh.load('../model/textured_cube.obj', process=False)
render_mesh = pyrender.Mesh.from_trimesh(mesh)
if not mesh.visual.material:
    print("no material")
else:
    print("material exists")
scene = pyrender.Scene()
scene.add(render_mesh)
# pyrender.Viewer(scene, use_raymond_lighting=True)

z_distance = 3
yfov = 2 * np.arctan(1 / z_distance)
camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.3333)  
camera_poses=np.load('camera_poses/camera_poses.npy')

light_poses=np.load('camera_poses/light_poses.npy')
# plot_all_poses(light_poses)
light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
for i, light_pose in enumerate(light_poses):
    scene.add(light, pose=light_pose)
scene.add(light, pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])

for i, camera_pose in enumerate(camera_poses):
   camera_node = scene.add(camera, pose=camera_pose)
   r = pyrender.OffscreenRenderer(800, 600)
   color, depth = r.render(scene)
   plt.imsave(f'../result/images/camera_{i}.png', color)
   plt.figure(figsize=(10, 5))
   plt.subplot(1, 2, 1)
   plt.imshow(color)
   plt.axis('off')
   plt.title('Color Image')

   plt.subplot(1, 2, 2)
   plt.imshow(depth, cmap='gray')
   plt.axis('off')
   plt.title('Depth Image')
   plt.show()

   scene.remove_node(camera_node)
   r.delete()



import numpy as np
import matplotlib.pyplot as plt

camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

cube_corners = np.array([
    [2, 2, 2],
    [2, 2, -2],
    [2, -2, 2],
    [2, -2, -2],
    [-2, 2, 2],
    [-2, 2, -2],
    [-2, -2, 2],
    [-2, -2, -2]
])

camera_poses = []

for corner in cube_corners:
    # point to origin
    z_axis = -corner / np.linalg.norm(corner)
    

    if np.allclose(z_axis, [0, 1, 0]):
        aux_vector = np.array([1, 0, 0])
    else:
        aux_vector = np.array([0, 1, 0])
    
    x_axis = np.cross(z_axis, aux_vector)
    x_axis /= np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    

    rotation_matrix = np.array([
        [x_axis[0], y_axis[0], z_axis[0], corner[0]],
        [x_axis[1], y_axis[1], z_axis[1], corner[1]],
        [x_axis[2], y_axis[2], z_axis[2], corner[2]],
        [0, 0, 0, 1]
    ])
    
    camera_poses.append(rotation_matrix)


camera_poses = np.array(camera_poses)
np.save('camera_poses/light_poses.npy', camera_poses)


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

plot_all_poses(camera_poses)
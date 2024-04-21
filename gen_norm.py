import torch
import numpy as np
from wis3d import Wis3D
from torchmetrics.functional.image import image_gradients

v3d = Wis3D("dbg", "vis_N", "xyz")

depth = np.load("output/0419-01352-2D-garden/train/ours_30000/depthfile/00000.npz")["depth"]    
depth = torch.tensor(depth, dtype=torch.float32, device="cuda")
dx, dy = image_gradients(depth.unsqueeze(0))
print(dx.shape, dy.shape)
dx = dx.squeeze(0).squeeze(0)
dy = dy.squeeze(0).squeeze(0)
grad_x = torch.stack([dx, torch.zeros_like(dx), torch.zeros_like(dx)], dim=0)
grad_y = torch.stack([torch.zeros_like(dy), dy, torch.zeros_like(dy)], dim=0)
normal = torch.cross(grad_x, grad_y, dim=0)
normal = normal / torch.norm(normal, dim=0, keepdim=True)


import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

'''
p 0.336173, 0.680479, 0.599159, rot: 1.000000, 0.000000, 0.000000, 0.000000, s: 0.024443, 0.024443 
pix.x: -0.008010, pix.y: -0.289274
view matrix: 0.885722, 0.184677, -0.425899, 0.000000, 
-0.233545, 0.970170, -0.065009, 0.000000, 
0.401189, 0.157047, 0.902432, 0.000000, 
-0.378750, -1.815881, 3.213168, 1.000000
'''

p = torch.tensor([0.336173, 0.680479, 0.599159]).requires_grad_(True)
rot = torch.tensor([1.000000, 0.000000, 0.000000, 0.000000]).requires_grad_(True)
s = torch.tensor([0.024443, 0.024443]).requires_grad_(True)
R_matrix = torch.zeros((3,3))
R_matrix[0,0] = 1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3])
R_matrix[0,1] = 2.0 * (rot[1] * rot[2] - rot[0] * rot[3])
R_matrix[0,2] = 2.0 * (rot[1] * rot[3] + rot[0] * rot[2])
R_matrix[1,0] = 2.0 * (rot[1] * rot[2] + rot[0] * rot[3])
R_matrix[1,1] = 1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3])
R_matrix[1,2] = 2.0 * (rot[2] * rot[3] - rot[0] * rot[1])
R_matrix[2,0] = 2.0 * (rot[1] * rot[3] - rot[0] * rot[2])
R_matrix[2,1] = 2.0 * (rot[2] * rot[3] + rot[0] * rot[1])
R_matrix[2,2] = 1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2])

tsu = s[0] * R_matrix @ torch.tensor([1.0,0.0,0.0])
tsv = s[1] * R_matrix @ torch.tensor([0.0,1.0,0.0])

view_matrix = torch.tensor([[0.885722, 0.184677, -0.425899, 0.000000],
                            [-0.233545, 0.970170, -0.065009, 0.000000],
                            [0.401189, 0.157047, 0.902432, 0.000000],
                            [-0.378750, -1.815881, 3.213168, 1.000000]])

R_cam = view_matrix[:3,:3].T
r1, r2, r3 = R_cam.T[0], R_cam.T[1], R_cam.T[2]
T_cam = view_matrix[3,:3]

A = torch.tensor([[0.021649, -0.005708, 0.000459],
[0.004514, 0.023713, -0.999521],
[-0.010410, -0.001589, 3.566455]]).requires_grad_(True)
# A[0,0] = r1 @ tsu
# A[0,1] = r1 @ tsv
# A[0,2] = r1 @ p + T_cam[0]
# A[1,0] = r2 @ tsu
# A[1,1] = r2 @ tsv
# A[1,2] = r2 @ p + T_cam[1]
# A[2,0] = r3 @ tsu
# A[2,1] = r3 @ tsv
# A[2,2] = r3 @ p + T_cam[2]

x, y = -0.008010, -0.289274

hu = A.T @ torch.tensor([-1, 0, x])
hv = A.T @ torch.tensor([0, -1, y])
hu.retain_grad()
hv.retain_grad()

denom = hu[0] * hv[1] - hu[1] * hv[0]

u = (hu[1] * hv[2] - hu[2] * hv[1]) / denom
v = (hu[2] * hv[0] - hu[0] * hv[2]) / denom


intersect_w = p + tsu * u + tsv * v
intersect_w = torch.cat([intersect_w, torch.tensor([1.0])])
intersect_c = (view_matrix.T @ intersect_w)[:3]

depth = intersect_c[-1]
print(depth)


depth.retain_grad()
p.retain_grad()
rot.retain_grad()
s.retain_grad()
A.retain_grad()

depth.backward()

print('depth ', depth.grad)
print('p ', p.grad)
print('rot ', rot.grad)
print('s ', s.grad)
print('A ', A.grad)

print(view_matrix[0,2], tsu[0], u, s[0])
print(view_matrix[1,2], tsu[1], u, s[0])
print(view_matrix[2,2], tsu[2], u, s[0])

print(hu, hv)
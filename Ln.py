import torch


with open("Ln.txt", "r") as f:
    data = f.readlines()

alphas = []
viewmat = []
rots = []
ray_dir = []
omegas_raw = []

for line in data:
    if "backray" in line:
        linedata = line.strip().split(';')
        for item in linedata:
            if "viewmat" in item:
                item = item.split('=')
                item = item[1].strip(' []').split(',')
                for i in range(len(item)):
                    viewmat.append(float(item[i]))
            if "ray_dir" in item:
                item = item.split('=')
                item = item[1].strip(' []').split(',')
                for i in range(len(item)):
                    ray_dir.append(float(item[i]))
    else:
        linedata = line.strip().split(',')
        quat = []
        for item in linedata:
            item = item.split('=')
            if item[0].strip() == 'omega':
                omegas_raw.append(float(item[1]))
            if item[0].strip() == 'alpha':
                alphas.append(float(item[1]))
            if item[0].strip() == 'r':
                quat.append(float(item[1]))
            if item[0].strip() == 'x':
                quat.append(float(item[1]))
            if item[0].strip() == 'y':
                quat.append(float(item[1]))
            if item[0].strip() == 'z':
                quat.append(float(item[1]))
        rots.append(quat)
rots = torch.tensor(rots, requires_grad=True, dtype=torch.float64)
viewmat = torch.tensor(viewmat, requires_grad=True, dtype=torch.float64).view(4, 4).T
alphas = alphas[::-1]
alphas = torch.tensor(alphas, requires_grad=True, dtype=torch.float64)
Ts = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.float64), 1.0 - alphas[:-1]]), dim=0)
omegas = Ts * alphas
omegas = omegas.flip(0)
# normal: the third column of the rotation matrix from the quaternion
normal_world = torch.zeros(rots.shape[0], 3, dtype=torch.float64)
normal_world[:, 0] = 2.0 * (rots[:, 1] * rots[:, 3] + rots[:, 0] * rots[:, 2])
normal_world[:, 1] = 2.0 * (rots[:, 2] * rots[:, 3] - rots[:, 0] * rots[:, 1])
normal_world[:, 2] = 1.0 - 2.0 * (rots[:, 1]**2 + rots[:, 2]**2)
normal_cam = torch.matmul(normal_world, viewmat[:3, :3].t())
N = torch.tensor([-0.4516,  0.5622, -0.6927], dtype=torch.float64, requires_grad=True) # [:, 480, 800]
ray_dir = torch.tensor(ray_dir, requires_grad=True, dtype=torch.float64).view(-1, 3)
# align the norm towards the ray
dot_prod = torch.sum(normal_cam * ray_dir, dim=1)
normal_cam[dot_prod > 0] *= -1

loss = (omegas * (1.0 - (normal_cam * N).sum(dim=1))).sum()
loss.backward()
print(rots.grad)
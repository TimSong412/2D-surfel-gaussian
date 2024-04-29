import torch


def ndc(z):
    far = 1000.0
    near = 0.2
    return (far+ near)/(far - near) - 2.0 * far * near / ((far - near) * z)

with open("ray.txt", "r") as f:
    data = f.readlines()

omegas = []
# ms = []
alphas = []
zs = []
for line in data:
    line = line.strip().split(',')
    for item in line:
        item = item.split('=')
        if item[0].strip() == 'omega':
            omegas.append(float(item[1]))
        if item[0].strip() == 'z':
            zs.append(float(item[1]))
        if item[0].strip() == 'alpha':
            alphas.append(float(item[1]))

omegas = omegas[::-1]
# ms = ms[::-1]
zs = zs[::-1]
alphas = alphas[::-1]

omegas_ref = torch.tensor(omegas, requires_grad=True, dtype=torch.float64)
zs = torch.tensor(zs, requires_grad=True, dtype=torch.float64)
alphas = torch.tensor(alphas, dtype=torch.float64, requires_grad=True)
Ts = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.float64), 1.0 - alphas[:-1]]), dim=0)
omegas = Ts * alphas
ms = ndc(zs)

L = torch.tensor([0.0], dtype=torch.float64)
for i in range(1, len(omegas)):
    for j in range(i):
        L += (omegas[i] * omegas[j]) * (ms[i] - ms[j])**2
torch.set_printoptions(precision=10)
L.backward()
# print(omegas.grad)
# print(ms.grad)
print(alphas.grad)
print(zs.grad)

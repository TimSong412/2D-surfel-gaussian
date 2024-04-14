import torch
import torch.nn as nn


class Help(nn.Module):
    def __init__(self):
        super(Help, self).__init__()

    def forward(self, rot, mod, scale, p_k, view_matrix):
        q = rot
        SELFr = q[:, 0]
        SELFx = q[:, 1]
        SELFy = q[:, 2]
        SELFz = q[:, 3]
        # q.requires_grad_()
        # q.retain_grad()
        # SELFr.requires_grad_()
        # SELFx.requires_grad_()
        # SELFy.requires_grad_()
        # SELFz.requires_grad_()
        # SELFr.retain_grad()
        # SELFx.retain_grad()
        # SELFy.retain_grad()
        # SELFz.retain_grad()

        SELFR = torch.tensor([
            [1. - 2. * (SELFy * SELFy + SELFz * SELFz), 2. * (SELFx * SELFy - SELFr * SELFz),
             2. * (SELFx * SELFz + SELFr * SELFy)],
            [2. * (SELFx * SELFy + SELFr * SELFz), 1. - 2. *
             (SELFx * SELFx + SELFz * SELFz), 2. * (SELFy * SELFz - SELFr * SELFx)],
            [2. * (SELFx * SELFz - SELFr * SELFy), 2. * (SELFy * SELFz + SELFr * SELFx), 1. - 2. * (SELFx * SELFx + SELFy * SELFy)]], requires_grad=True)
        # SELFR.requires_grad_()
        # SELFR.retain_grad()
        # SELFSTuv = torch.zeros(6)

        # SELFR.requires_grad_()
        # SELFR.retain_grad()

        # SELFSTuv[0] = mod * scale[0] * SELFR[0][0]
        # SELFSTuv[1] = mod * scale[0] * SELFR[1][0]
        # SELFSTuv[2] = mod * scale[0] * SELFR[2][0]
        # SELFSTuv[3] = mod * scale[1] * SELFR[0][1]
        # SELFSTuv[4] = mod * scale[1] * SELFR[1][1]
        # SELFSTuv[5] = mod * scale[1] * SELFR[2][1]
        SELFSTuv = torch.tensor([
            mod * scale[:, 0] * SELFR[0][0],
            mod * scale[:, 0] * SELFR[1][0],
            mod * scale[:, 0] * SELFR[2][0],
            mod * scale[:, 1] * SELFR[0][1],
            mod * scale[:, 1] * SELFR[1][1],
            mod * scale[:, 1] * SELFR[2][1]
        ], requires_grad=True)
        SELFSTuv.retain_grad()

        # SELFSTuv.requires_grad_()
        # SELFSTuv.retain_grad()

        SELFr1s1 = view_matrix[:, 0] * SELFSTuv[:, 0] + view_matrix[:, 4] * \
            SELFSTuv[:, 1] + view_matrix[:, 8] * SELFSTuv[:, 2]
        SELFr1s2 = view_matrix[:, 0] * SELFSTuv[:, 3] + view_matrix[:, 4] * \
            SELFSTuv[:, 4] + view_matrix[:, 8] * SELFSTuv[:, 5]
        SELFr2s1 = view_matrix[:, 1] * SELFSTuv[:, 0] + view_matrix[:, 5] * \
            SELFSTuv[:, 1] + view_matrix[:, 9] * SELFSTuv[:, 2]
        SELFr2s2 = view_matrix[:, 1] * SELFSTuv[:, 3] + view_matrix[:, 5] * \
            SELFSTuv[:, 4] + view_matrix[:, 9] * SELFSTuv[:, 5]
        SELFr3s1 = view_matrix[:, 2] * SELFSTuv[:, 0] + view_matrix[:, 6] * \
            SELFSTuv[:, 1] + view_matrix[:, 10] * SELFSTuv[:, 2]
        SELFr3s2 = view_matrix[:, 2] * SELFSTuv[:, 3] + view_matrix[:, 6] * \
            SELFSTuv[:, 4] + view_matrix[:, 10] * SELFSTuv[:, 5]
        SELFr1p_t1 = view_matrix[:, 0] * p_k[:, 0] + view_matrix[:, 4] * \
            p_k[:, 1] + view_matrix[:, 8] * p_k[:, 2] + view_matrix[:, 12]
        SELFr2p_t2 = view_matrix[:, 1] * p_k[:, 0] + view_matrix[:, 5] * \
            p_k[:, 1] + view_matrix[:, 9] * p_k[:, 2] + view_matrix[:, 13]
        SELFr3p_t3 = view_matrix[:, 2] * p_k[:, 0] + view_matrix[:, 6] * \
            p_k[:, 1] + view_matrix[:, 10] * p_k[:, 2] + view_matrix[:, 14]

        # SELFr1s1.requires_grad_()
        # SELFr1s2.requires_grad_()
        # SELFr2s1.requires_grad_()
        # SELFr2s2.requires_grad_()
        # SELFr3s1.requires_grad_()
        # SELFr3s2.requires_grad_()
        # SELFr1p_t1.requires_grad_()
        # SELFr2p_t2.requires_grad_()
        # SELFr3p_t3.requires_grad_()
        # SELFr1s1.retain_grad()
        # SELFr1s2.retain_grad()
        # SELFr2s1.retain_grad()
        # SELFr2s2.retain_grad()
        # SELFr3s1.retain_grad()
        # SELFr3s2.retain_grad()
        # SELFr1p_t1.retain_grad()
        # SELFr2p_t2.retain_grad()
        # SELFr3p_t3.retain_grad()

        A = torch.tensor([SELFr1s1, SELFr1s2, SELFr1p_t1, SELFr2s1, SELFr2s2,
                         SELFr2p_t2, SELFr3s1, SELFr3s2, SELFr3p_t3], requires_grad=True)

        A.retain_grad()

        return A


def forw(rot, mod, scale, p_k, view_matrix):
    q = rot
    SELFr = q[0]
    SELFx = q[1]
    SELFy = q[2]
    SELFz = q[3]

    # SELFr.requires_grad_()
    # SELFx.requires_grad_()
    # SELFy.requires_grad_()
    # SELFz.requires_grad_()
    # SELFr.retain_grad()
    # SELFx.retain_grad()
    # SELFy.retain_grad()
    # SELFz.retain_grad()

    SELFR = torch.tensor([
        [1. - 2. * (SELFy * SELFy + SELFz * SELFz), 2. * (SELFx * SELFy - SELFr * SELFz),
            2. * (SELFx * SELFz + SELFr * SELFy)],
        [2. * (SELFx * SELFy + SELFr * SELFz), 1. - 2. *
            (SELFx * SELFx + SELFz * SELFz), 2. * (SELFy * SELFz - SELFr * SELFx)],
        [2. * (SELFx * SELFz - SELFr * SELFy), 2. * (SELFy * SELFz + SELFr * SELFx), 1. - 2. * (SELFx * SELFx + SELFy * SELFy)]], requires_grad=True)
    # SELFSTuv = torch.zeros(6)

    # SELFR.requires_grad_()
    # SELFR.retain_grad()

    # SELFSTuv[0] = mod * scale[0] * SELFR[0][0]
    # SELFSTuv[1] = mod * scale[0] * SELFR[1][0]
    # SELFSTuv[2] = mod * scale[0] * SELFR[2][0]
    # SELFSTuv[3] = mod * scale[1] * SELFR[0][1]
    # SELFSTuv[4] = mod * scale[1] * SELFR[1][1]
    # SELFSTuv[5] = mod * scale[1] * SELFR[2][1]
    SELFSTuv = torch.tensor([
        mod * scale[0] * SELFR[0][0],
        mod * scale[0] * SELFR[1][0],
        mod * scale[0] * SELFR[2][0],
        mod * scale[1] * SELFR[0][1],
        mod * scale[1] * SELFR[1][1],
        mod * scale[1] * SELFR[2][1]
    ], requires_grad=True)

    # SELFSTuv.requires_grad_()
    # SELFSTuv.retain_grad()

    SELFr1s1 = view_matrix[0] * SELFSTuv[0] + view_matrix[4] * \
        SELFSTuv[1] + view_matrix[8] * SELFSTuv[2]
    SELFr1s2 = view_matrix[0] * SELFSTuv[3] + view_matrix[4] * \
        SELFSTuv[4] + view_matrix[8] * SELFSTuv[5]
    SELFr2s1 = view_matrix[1] * SELFSTuv[0] + view_matrix[5] * \
        SELFSTuv[1] + view_matrix[9] * SELFSTuv[2]
    SELFr2s2 = view_matrix[1] * SELFSTuv[3] + view_matrix[5] * \
        SELFSTuv[4] + view_matrix[9] * SELFSTuv[5]
    SELFr3s1 = view_matrix[2] * SELFSTuv[0] + view_matrix[6] * \
        SELFSTuv[1] + view_matrix[10] * SELFSTuv[2]
    SELFr3s2 = view_matrix[2] * SELFSTuv[3] + view_matrix[6] * \
        SELFSTuv[4] + view_matrix[10] * SELFSTuv[5]
    SELFr1p_t1 = view_matrix[0] * p_k[0] + view_matrix[4] * \
        p_k[1] + view_matrix[8] * p_k[2] + view_matrix[12]
    SELFr2p_t2 = view_matrix[1] * p_k[0] + view_matrix[5] * \
        p_k[1] + view_matrix[9] * p_k[2] + view_matrix[13]
    SELFr3p_t3 = view_matrix[2] * p_k[0] + view_matrix[6] * \
        p_k[1] + view_matrix[10] * p_k[2] + view_matrix[14]

    # SELFr1s1.requires_grad_()
    # SELFr1s2.requires_grad_()
    # SELFr2s1.requires_grad_()
    # SELFr2s2.requires_grad_()
    # SELFr3s1.requires_grad_()
    # SELFr3s2.requires_grad_()
    # SELFr1p_t1.requires_grad_()
    # SELFr2p_t2.requires_grad_()
    # SELFr3p_t3.requires_grad_()
    # SELFr1s1.retain_grad()
    # SELFr1s2.retain_grad()
    # SELFr2s1.retain_grad()
    # SELFr2s2.retain_grad()
    # SELFr3s1.retain_grad()
    # SELFr3s2.retain_grad()
    # SELFr1p_t1.retain_grad()
    # SELFr2p_t2.retain_grad()
    # SELFr3p_t3.retain_grad()

    A = torch.tensor([SELFr1s1, SELFr1s2, SELFr1p_t1, SELFr2s1, SELFr2s2,
                     SELFr2p_t2, SELFr3s1, SELFr3s2, SELFr3p_t3], requires_grad=True)

    # A.retain_grad()

    return A


if __name__ == "__main__":
    rot = torch.tensor([0.999675691, 0.00921958871, 0.00530451629, 0.0231338646], requires_grad=True)

    mod = torch.tensor(1.0, requires_grad=True)
    scale = torch.tensor([0.0633356348, 0.046614103], requires_grad=True)
    p_k = torch.tensor([5.37206268,
                        1.64806199, 
                        -0.650559425], requires_grad=True)
    view_matrix = torch.tensor([[-0.68113625, -0.434350938, 0.589400291, 0],
                                [0.322851121, 0.544342637, 0.774246871, 0],
                                [-0.657130599, 0.717656136, -0.230540797, 0],
                                [0.473002672, -1.44897664, 3.07225776, 1]]).reshape(-1)

    q = rot
    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    r.retain_grad()
    x.retain_grad()
    y.retain_grad()
    z.retain_grad()

    R0 = torch.stack([1. - 2. * (y * y + z * z), 2. *
                     (x * y - r * z), 2. * (x * z + r * y)])
    R1 = torch.stack([2. * (x * y + r * z), 1. - 2. *
                     (x * x + z * z), 2. * (y * z - r * x)])
    R2 = torch.stack([2. * (x * z - r * y), 2. *
                     (y * z + r * x), 1. - 2. * (x * x + y * y)])

    R = torch.stack([R0, R1, R2])
    R.retain_grad()

    STuv = torch.stack([
        mod * scale[0] * R[0][0],
        mod * scale[0] * R[1][0],
        mod * scale[0] * R[2][0],
        mod * scale[1] * R[0][1],
        mod * scale[1] * R[1][1],
        mod * scale[1] * R[2][1]
    ])

    STuv.retain_grad()

    r1s1 = view_matrix[0] * STuv[0] + view_matrix[4] * \
        STuv[1] + view_matrix[8] * STuv[2]
    r1s2 = view_matrix[0] * STuv[3] + view_matrix[4] * \
        STuv[4] + view_matrix[8] * STuv[5]
    r2s1 = view_matrix[1] * STuv[0] + view_matrix[5] * \
        STuv[1] + view_matrix[9] * STuv[2]
    r2s2 = view_matrix[1] * STuv[3] + view_matrix[5] * \
        STuv[4] + view_matrix[9] * STuv[5]
    r3s1 = view_matrix[2] * STuv[0] + view_matrix[6] * \
        STuv[1] + view_matrix[10] * STuv[2]
    r3s2 = view_matrix[2] * STuv[3] + view_matrix[6] * \
        STuv[4] + view_matrix[10] * STuv[5]

    r1p_t1 = view_matrix[0] * p_k[0] + view_matrix[4] * \
        p_k[1] + view_matrix[8] * p_k[2] + view_matrix[12]
    r2p_t2 = view_matrix[1] * p_k[0] + view_matrix[5] * \
        p_k[1] + view_matrix[9] * p_k[2] + view_matrix[13]
    r3p_t3 = view_matrix[2] * p_k[0] + view_matrix[6] * \
        p_k[1] + view_matrix[10] * p_k[2] + view_matrix[14]

    r1s1.retain_grad()
    r1s2.retain_grad()
    r2s1.retain_grad()
    r2s2.retain_grad()
    r3s1.retain_grad()
    r3s2.retain_grad()
    r1p_t1.retain_grad()
    r2p_t2.retain_grad()
    r3p_t3.retain_grad()

    A = torch.stack([r1s1,  r1s2,  r1p_t1,  r2s1,  r2s2,  r2p_t2,
                    r3s1,  r3s2,  r3p_t3])  # , requires_grad=True)
    A.retain_grad()

    # , requires_grad=True)
    dA = torch.tensor([1.69617506e-05, 
                       1.04270039e-05, 
                       -9.21413721e-06, 
                       -1.72967339e-05, 
                       -3.50973664e-06, 
                       6.00056228e-06, 
                       -2.61098944e-06,
                       1.43695911e-06,
                       9.91363791e-09])
    A.backward(dA)

    print(A.grad)
    print(rot.grad)

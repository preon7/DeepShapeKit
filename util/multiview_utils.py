"""
functions for multiview output from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""
# import trimesh
import yaml
import os
import numpy as np
import torch
import util.constants as c
import cv2

# from .renderer import Renderer
from .geometry import perspective_projection, perspective_projection_homo, perspective_projection_ref


def get_fullsize_masks(masks, bboxes, h=368, w=368):
    full_masks = []
    for i in range(len(masks)):
        box = bboxes[i]
        full_mask = torch.zeros([h, w], dtype=torch.bool)
        full_mask[box[1]:box[1] + box[3] + 1, box[0]:box[0] + box[2] + 1] = masks[i]
        full_masks.append(full_mask)
    full_masks = torch.stack(full_masks)

    return full_masks


def get_cam(device='cpu'):
    proj_m_set = torch.stack([c.proj_front, c.proj_bottom], 0).to(device)
    proj_m_set_homo = torch.cat([proj_m_set, torch.tensor([[[0,0,-1,0]], [[0,0,-1,0]]]).to(device)], 1)
    f1 = 3930.0
    f2 = 3930
    focal = torch.tensor([f1, f1]).to(device)
    center = torch.tensor([[1024.,520.],[1024.,520.]]).to(device)
    # K = torch.tensor([[f1/2048,0,0.],[0,f1/1040,0.],[0,0,1]]).to(device)
    K = torch.tensor([[f1, 0, 1024.], [0, f1, 520.], [0, 0, 1]]).to(device)
    # K = torch.tensor([[ 7.67578125,0.,-1.,0.],
    #                  [ 0.,15.11538462,1.,0.],
    #                  [ 0.,0.,-1.00010001,-0.100005],
    #                  [ 0.,0.,-1.,0.]])
    # K = torch.tensor([[3.83789062,0.,0.,0.],
    #                 [0.,7.55769231,0.,0.],
    #                 [0.,0.,- 1.00010001,- 0.100005],
    #                 [0.,0.,- 1.,0.]])
    H = torch.matmul(K.inverse(), proj_m_set)
    # H = torch.matmul(K.inverse(), proj_m_set_homo)

    distortion = torch.tensor(c.distortion).to(device)

    return  proj_m_set, focal, center, H[:,:,:-1], H[:,:,-1], distortion


def projection_loss(x, y):
    loss = (x.float() - y.float()).norm(p=2)
    return loss


def triangulation_LBFGS(x, proj_m, #rotation, camera_t, focal_length, camera_center,
                        distortion=None, device='cpu'):
    n = x.shape[0]
    X = torch.tensor([2.5, 1.2, 1.95])[None, None, :]
    X.requires_grad_()

    x = x.to(device)
    X = X.to(device)

    losses = []
    optimizer = torch.optim.LBFGS([X], lr=1, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        # projected_points = perspective_projection_ref(X.repeat(n, 1, 1), rotation, camera_t, focal_length, camera_center, distortion)
        projected_points = perspective_projection(X.repeat(n, 1, 1), proj_m)
        loss = projection_loss(projected_points.squeeze(), x)

        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        # projected_points = perspective_projection_ref(X.repeat(n, 1, 1), rotation, camera_t, focal_length, camera_center, distortion)
        projected_points = perspective_projection(X.repeat(n, 1, 1), proj_m)
        loss = projection_loss(projected_points.squeeze(), x)
        losses.append(loss.detach().item())
    X = X.detach().squeeze()

    return X, losses


def triangulation(x, proj_m, #rotation, camera_t, focal_length, camera_center,
                  distortion=None, device='cpu'):
    n = x.shape[0]
    X = torch.tensor([2.5, 1.2, 1.95])[None, None, :]
    X.requires_grad_()

    x = x.to(device)
    X = X.to(device)

    losses = []
    optimizer = torch.optim.Adam([X], lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 90], gamma=0.1)
    for i in range(100):
        # projected_points = perspective_projection_ref(X.repeat(n, 1, 1), rotation, camera_t, focal_length, camera_center, distortion)
        projected_points = perspective_projection(X.repeat(n, 1, 1), proj_m)
        loss = projection_loss(projected_points.squeeze(), x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().item())

    X = X.detach().squeeze()

    return X, losses


def get_gt_3d(keypoints, frames, LBFGS=False):
    '''
    Input:
        keypoints (bn, kn, 2): 2D kpts from different views
        frames (bn): frame numbers
    Output:
        kpts_3d (kn, 4): ground truth 3D kpts, with validility
    '''
    bn, kn, _ = keypoints.shape
    kpts_3d = torch.zeros([kn, 4])

    #
    proj_m_set, focal, center, R, T, distortion = get_cam()
    kpts_valid = []
    cams = []
    for i in range(kn):
        valid = keypoints[:, i, -1] > 0
        kpts_valid.append(keypoints[valid, i, :2])
        cams.append(valid)

    #
    for i in range(kn):
        x = kpts_valid[i]
        if len(x) >= 2:
            # proj_m = proj_m_set

            if LBFGS:
                X, _ = triangulation_LBFGS(x, proj_m_set)
            else:
                X, _ = triangulation(x, proj_m_set)

            kpts_3d[i, :3] = X
            kpts_3d[i, -1] = 1

    return kpts_3d


def Procrustes(X, Y):
    """
    Solve full Procrustes: Y = s*RX + t
    Input:
        X (N,3): tensor of N points
        Y (N,3): tensor of N points in world coordinate
    Returns:
        R (3x3): tensor describing camera orientation in the world (R_wc)
        t (3,): tensor describing camera translation in the world (t_wc)
        s (1): scale
    """
    # remove translation
    A = (Y - Y.mean(dim=0, keepdim=True))
    B = (X - X.mean(dim=0, keepdim=True))

    # remove scale
    sA = (A * A).sum() / A.shape[0]
    sA = sA.sqrt()
    sB = (B * B).sum() / B.shape[0]
    sB = sB.sqrt()
    A = A / sA
    B = B / sB
    s = sA / sB

    # to numpy, then solve for R
    A = A.t().numpy()
    B = B.t().numpy()

    M = B @ A.T
    U, S, VT = np.linalg.svd(M)
    V = VT.T

    d = np.eye(3)
    d[-1, -1] = np.linalg.det(V @ U.T)
    R = V @ d @ U.T

    # back to tensor
    R = torch.tensor(R).float()
    t = Y.mean(axis=0) - R @ X.mean(axis=0) * s

    return R, t, s

def render_vertex_on_frame(img, vertex_posed, fish, frame, kpts=None, bboxs=None):
    proj_m_set, focal, center, R, T, distortion = get_cam()

    points = torch.einsum('bij,bkj->bki', R[frame], vertex_posed[0]) + T[frame]
    points = points[0]
    # rot = np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
    rot = np.eye(3)
    # print('middle point: {}'.format(torch.sum(points, axis=-2) / points.size(-2)))

    # Rendering
    bg_img = np.zeros((1040,2048,3))
    bg_img[:img.shape[0], :img.shape[1],:] = torch.tensor(img)

    # manual transform
    # projected = perspective_projection(vertex_posed[0].repeat(2,1,1), proj_m_set)[ frame[0]]
    projected_ref = perspective_projection_ref(vertex_posed[0].repeat(2,1,1), R, T, focal, center)[ frame[0]]
    ix = (torch.minimum(torch.maximum(projected_ref[:, 1].int(), torch.tensor(0)), torch.tensor(1039))).tolist()
    iy = (torch.minimum(torch.maximum(projected_ref[:, 0].int(), torch.tensor(0)), torch.tensor(2047))).tolist()
    img[ix, iy] = np.array([0, 255, 0])

    # visualize keypoints
    if kpts is not None:
        for i in range(kpts[0].size(1)):
            img[int(kpts[0][frame[0],i,1]), int(kpts[0][frame[0],i,0])] = np.array([255, 0, 0])
        # for i in range(kpts[1].size(1)):
        #     img[int(kpts[1][frame[0],i,1]), int(kpts[1][frame[0],i,0])] = np.array([255, 125, 0])

    if bboxs is not None:
        #print('draw bbox')
        img[bboxs[1].item():bboxs[3].item(), bboxs[0].item()] = np.array([255, 255, 0])
        img[bboxs[1].item():bboxs[3].item(), bboxs[2].item()] = np.array([255, 255, 0])
        img[bboxs[1].item(), bboxs[0].item():bboxs[2].item()] = np.array([255, 255, 0])
        img[bboxs[3].item(), bboxs[0].item():bboxs[2].item()] = np.array([255, 255, 0])

    img_pose = img.astype(np.uint8)

    return img_pose, img_pose
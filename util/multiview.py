"""
functions for multiview modified from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""

import cv2
import numpy as np
import torch

from model.losses import camera_fitting_loss
# from util.renderer import Renderer
from util.geometry import perspective_projection
import util.multiview_utils as mutils
import util.constants as c


def fit_geometry(fish, keypoints, frames, init_pose=None, init_bone=None):
    # 3D kpts on fish
    if init_pose == None and init_bone == None:
        fish_kpts = torch.matmul(fish.vert2kpt, fish.V[0])
    else:
        init_ori = torch.zeros([1, 3]).float()

        fish_output = fish(init_ori, init_pose, init_bone)
        fish_kpts = fish_output['keypoints'][0]

    # Triangulation with LBFGS
    kpts_3d = mutils.get_gt_3d(keypoints, frames, LBFGS=True)

    valid_3d = kpts_3d[:, -1] > 0
    valid_kpts_3d = kpts_3d[valid_3d, :3]
    fish_kpts = fish_kpts[valid_3d, :]

    # Procrustes with available 3D kpts
    R, t, s = mutils.Procrustes(fish_kpts, valid_kpts_3d)
    aa, _ = cv2.Rodrigues(R.numpy())

    init_ori = torch.tensor(aa).reshape(1, 3).float()
    init_t = t
    init_s = s

    return init_ori, init_t, init_s


def fit_mesh(fish, optimizer, keypoints, frames, masks, renderer, device='cpu',
             init_ori=None, init_t=None, init_s=None, init_pose=None, init_bone=None, img_filenames=None, index=None, bboxs=None):
    '''
    Only used in multiview and crossview fitting:
    Input:
        init_pose (vn, 4*3): body pose in axis-angle (exclude root joint orient)
        init_bone (vn, 4): bone length
    '''

    if init_ori != None or init_t != None or init_s != None or init_pose !=None or init_bone != None:
        has_prev = True
        #optimizer.prior_weight = 80
    else:
        has_prev = False

    ### Triangulation + Procrustes as initialization
    if init_ori==None and init_t==None and init_s==None:
        init_ori, init_t, init_s = fit_geometry(fish, keypoints, frames,
                                                init_pose=init_pose, init_bone=init_bone)

    ### If not provided (as in multiview), initialize with canonical
    if init_pose is None:
        init_pose = torch.zeros([1, c.num_bone * 3])
    if init_bone is None:
        init_bone = torch.ones([1, c.num_bone])

    #### Change suitable format for optimizer
    ###### particularly, combine orient and body pose
    init_pose = torch.cat([init_ori, init_pose], dim=1)
    init_pose = init_pose.float().to(device)
    init_bone = init_bone.float().to(device)
    init_s = init_s.float().to(device)
    init_t = init_t.float().to(device)

    ### Camera parameter
    proj_m_set, focal, center, R, T, distortion = mutils.get_cam()
    cam_rot = R.to(device)
    cam_t = T.to(device)
    focal = focal.to(device)
    center = center.to(device)

    ### Mesh fitting
    keypoints = keypoints.to(device)
    vertices, pose_est, bone_est, scale_est, t, losses = optimizer(init_pose, init_bone,
                                                           init_t, init_s,
                                                           proj_m_set, keypoints, masks, renderer, has_prev, img_filenames, index, bboxs)

    ### Generating mesh output
    fish_output = fish(pose_est[:, 0:3],
                       pose_est[:, 3:],
                       bone_est, scale_est)

    vertex_posed = fish_output['vertices'] + t
    mesh_keypoint = fish_output['keypoints'] + t

    return vertex_posed, mesh_keypoint, t, pose_est, bone_est, scale_est, losses


def multiview_rigid_alignment(fish, pose, bone, keypoints, frames, device='cpu', num_iters=100):
    '''
    Rigidly align single view reconstruction to multiview instance so we can
    check reconstruction accuracy across different views.
    1. First run general Procrustes for global alignment
    2. Because Procrustes can only use keypoints that are visible from at least two views,
    we run a short optimizition (rigidly, fixed pose and shape) afterward to improve alignment.
    Input:
        pose and bone are from singleview reconstruction;
        keypoints are ground truth for alignments
    '''

    ### Camera parameter
    proj_m_set, focal, center, R, T, distortion = mutils.get_cam(device)

    ### Triangulation + Procrustes for global alignemnt
    global_orient, global_t, scale = fit_geometry(fish, keypoints, frames,
                                                  init_pose=pose, init_bone=bone)

    ### Optimization to improve alignment
    pose = pose.detach().clone().to(device)
    bone = bone.detach().clone().to(device)
    keypoints = keypoints.clone().to(device)
    batch_size = len(frames)

    global_orient = global_orient.to(device)
    global_t = global_t.to(device)
    scale = scale.to(device)

    global_orient.requires_grad = True
    global_t.requires_grad = True
    scale.requires_grad = True

    global_params = [global_orient, global_t, scale]
    global_optimizer = torch.optim.Adam(global_params, lr=1e-2, betas=(0.9, 0.999))
    for i in range(num_iters):
        fish_output = fish(global_pose=global_orient,
                           body_pose=pose,
                           bone_length=bone,
                           scale=scale)

        model_keypoints = fish_output['keypoints'] + global_t.repeat(1, 1, 1)
        model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

        loss = camera_fitting_loss(model_keypoints, proj_m_set,
                                   keypoints[:, :, :2], keypoints[:, :, -1])

        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()

    # Output
    fish_output = fish(global_pose=global_orient,
                       body_pose=pose,
                       bone_length=bone,
                       scale=scale)
    model_mesh = fish_output['vertices'] + global_t.repeat(1, 1, 1)
    model_keypoints = fish_output['keypoints'] + global_t.repeat(1, 1, 1)

    model_mesh = model_mesh.detach().to('cpu')
    model_keypoints = model_keypoints.detach().to('cpu')

    return model_mesh, model_keypoints


def reproject_masks(vertex_est, renderer_list, frames):
    # Transform vertex for each camera view
    proj_m_set, focal, center, R, T, distortion = mutils.get_cam()
    rotation = torch.stack([R,R],0)
    translation = torch.stack([T,T],0).unsqueeze(1)

    points = vertex_est.repeat([len(frames), 1, 1])
    points = torch.einsum('bij,bkj->bki', rotation, points) + translation
    # Render for each view
    img = torch.zeros([368, 368, 3])
    proj_masks = []
    for i in range(len(frames)):
        renderer = renderer_list[frames[i]]
        img_pose, depth_map = renderer(points[i].cpu().numpy(), np.eye(3), [0, 0, 0], img.clone().numpy())
        mask = torch.tensor(depth_map > 0)
        proj_masks.append(mask)

    return proj_masks


def reproject_keypoints(mesh_keypoints, frames):
    proj_m_set, focal, center, R, T, distortion = mutils.get_cam()

    kpts = mesh_keypoints.repeat([len(frames), 1, 1])
    proj_kpts = perspective_projection(kpts, proj_m_set)

    return proj_kpts
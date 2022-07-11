"""
linear blend skinning from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""

import torch
from torch.nn import functional as F

from util.geometry import batch_rodrigues


class LBS():
    '''
    Implementation of linear blend skinning, with additional bone and scale
    Input:
        V (BN, V, 3): vertices to pose and shape
        pose (BN, J, 3, 3) or (BN, J, 3): pose in rot or axis-angle
        bone (BN, K): allow for direct change of relative joint distances
        scale (1): scale the whole kinematic tree
    '''

    def __init__(self, J, parents, weights):
        self.n_joints = J.shape[1] - 1
        # rotation around joint
        # self.h_joints = F.pad(J[:,1:].unsqueeze(-1), [0, 0, 0, 1], value=0)
        # #self.kin_tree = torch.cat([J[:, [0], :], J[:, 1:] - J[:, parents[1:]]], dim=1).unsqueeze(-1)
        # self.kin_tree = (J[:, 1:] - J[:, parents[1:]]).unsqueeze(-1)

        # rotation around parent
        self.h_joints = F.pad((J[:, parents[1:]]).unsqueeze(-1), [0, 0, 0, 1], value=0)
        self.kin_tree = (J[:, 1:] - J[:, parents[1:]]).unsqueeze(-1)
        self.joint_parent = F.pad((J[:, parents[1:]]).unsqueeze(-1), [0,0,0,1], value=1)

        self.parents = parents
        #self.parents[0] = -1
        self.weights = weights[None].float()

    def __call__(self, V, pose, bone, scale, to_rotmats=True):
        batch_size = len(V)
        device = pose.device
        V = F.pad(V.unsqueeze(-1), [0, 0, 0, 1], value=1)
        kin_tree = (scale * self.kin_tree) * bone[:, 1:, None, None]

        # disable rotation around x-axis for fish parts
        for i in range(self.n_joints - 1):
            pose[0,3 * (i + 1)] = 0.

        if to_rotmats:
            pose = batch_rodrigues(pose.view(-1, 3))
        pose = pose.view([batch_size, -1, 3, 3])
        T = torch.zeros([batch_size, self.n_joints, 4, 4]).float().to(device)
        T[:, :, -1, -1] = 1
        T[:, :, :3, :] = torch.cat([pose[:,1:,:,:], kin_tree], dim=-1)
        T_rel = [T[:, 0]]
        for i in range(1, self.n_joints):
            T_rel.append(T_rel[self.parents[i]] @ T[:, i])
        T_rel = torch.stack(T_rel, dim=1)
        T_rel[:, :, :, [-1]] -= T_rel.clone() @ (self.h_joints * scale)

        # rotation around parent
        # parent_pos = T_rel.clone() @ (self.joint_parent * scale) - T_rel[:, [0,0,1,2,3], :, :].clone() @ (self.joint_parent * scale)
        # parent_pos[:,1:,:,:] += parent_pos[:,[0],:,:]
        # parent_pos[:, 2:, :, :] += parent_pos[:, [1], :, :]
        # parent_pos[:, 3:, :, :] += parent_pos[:, [2], :, :]
        # parent_pos[:, 4:, :, :] += parent_pos[:, [3], :, :]
        # T_rel[:, :, :, [-1]] += parent_pos

        T_ = self.weights @ T_rel.view(batch_size, self.n_joints, -1)
        T_ = T_.view(batch_size, -1, 4, 4)

        V = T_ @ V

        R = torch.zeros([batch_size, 1, 4, 4]).float().to(device)
        R[:, :, :3, :3] = pose[:,[0],:,:]
        R[:, :, -1, -1] = 1
        V = R @ V

        return V[:, :, :3, 0] / V[:, :, [3], 0]

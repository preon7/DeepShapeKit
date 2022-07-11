"""
load articulated fish model from json file
modified from bird model by Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""

import os
import json
import torch
from .LBS import LBS
#from .LBS_origin import LBS


class fish_model():

    def __init__(self, device=torch.device('cpu'), mesh='goldfish_design_small.json'):
        self.device = device

        # read in fish model from the same dir
        this_dir = os.path.dirname(__file__)
        mesh_file = os.path.join(this_dir, mesh)
        with open(mesh_file, 'r') as infile:
            dd = json.load(infile)

        self.dd = dd

        # triangulate if input mesh has quad faces
        # fish_faces = dd['F']
        # triangle_faces = []
        # for face in fish_faces:
        #     if len(face) == 4:
        #         triangle_faces.append([face[0], face[1], face[2]])
        #         triangle_faces.append([face[2], face[3], face[0]])
        #     elif len(face) == 3:
        #         triangle_faces.append(face)
        #     else:
        #         raise Exception('face data incorrect: got vertices number not in [3,4]')
        # self.faces = torch.tensor(triangle_faces).to(device)

        self.faces = torch.tensor(dd['F'])

        self.kintree_table = torch.tensor(dd['kintree_table'])[:,:].to(device)
        self.parents = self.kintree_table[0][:]

        self.weights = torch.tensor(dd['weights']).to(device)
        self.vert2kpt = torch.tensor(dd['vert2kpt']).to(device)

        self.J = torch.tensor(dd['J']).unsqueeze(0).to(device)

        self.V = torch.tensor(dd['V']).unsqueeze(0).to(device)
        self.V = self.V - self.J[0,0]
        self.J = self.J - self.J[0,0]

        self.V = self.V * 0.01
        self.J = self.J * 0.01

        self.LBS = LBS(self.J, self.parents, self.weights)

    def __call__(self, global_pose, body_pose, bone_length, scale=1, pose2rot=True):
        global_pose = global_pose.to(self.device)
        body_pose = body_pose.to(self.device)
        bone_length = bone_length.to(self.device)
        batch_size = global_pose.shape[0]
        V = self.V.repeat([batch_size, 1, 1]) * scale

        # concatenate bone and pose
        bone = torch.cat([torch.ones([batch_size, 1]).to(self.device), bone_length], dim=1)
        pose = torch.cat([global_pose, body_pose], dim=1)

        # LBS
        verts = self.LBS(V, pose, bone, scale, to_rotmats=pose2rot)

        # Calculate 3d keypoint from new vertices resulted from pose
        keypoints = []
        for i in range(verts.shape[0]):
            kpt = torch.matmul(self.vert2kpt, verts[i])
            keypoints.append(kpt)
        keypoints = torch.stack(keypoints)

        # Final output after articulation
        output = {'vertices': verts,
                  'keypoints': keypoints}

        return output

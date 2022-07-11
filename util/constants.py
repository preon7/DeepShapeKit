"""
parameters and constrains, modified from Badger et al.
@Inproceedings{badger2020,
  Title          = {3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View},
  Author         = {Badger, Marc and Wang, Yufu and Modh, Adarsh and Perkes, Ammon and Kolotouros, Nikos and Pfrommer, Bernd and Schmidt, Marc and Daniilidis, Kostas},
  Booktitle      = {ECCV},
  Year           = {2020}
}
https://github.com/marcbadger/avian-mesh
"""

import torch

in_size = (2048,1040)
kpt_map = {
    0: 'head',
    1: 'gill',
    2: 'tail',
    3: 'tail_tip',
    4: 'belly'
    }

# Camera
# proj_front = torch.tensor([[-0.173732177315659, 0.707692897282493, -0.00379997317811881, 0.0260513031101779],
#                            [-0.0653038990589222, 0.00430721305649784, -0.653343935849277, 0.192751694352691],
#                            [-0.000162547056514521, 1.70570238833904e-05, 4.64850171959407e-06, 0.000302115230578408]])

# undistorted with [-0.568857779226978,0.151730496415158, 0, 0, 0]
# proj_front = torch.tensor([[0.195300086254197,	-0.687445399864040,	0.0142533660536701,	-0.0328436914058996], # demo_2021
#                             [0.0769391282586720,	0.00605248614806819,	0.665551600718732,	-0.197685370913989],
#                             [0.000184153466409749,	-7.32110858327708e-08,	5.89051142009833e-06,	-0.000311320093050174]])
proj_front = torch.tensor([[0.195699896031463,	-0.685721414161406,	0.0150748084109422,	-0.0232585284611645],  # 20171216_122522
                            [0.0774615176058445,	0.00632549766348506,	0.668022328306539,	-0.196023539952932],
                            [0.000183019259769104,	-4.82734212940345e-08,	6.40608374020802e-06,	-0.000303907225124089]])

intrinsic_mat = [[3946, 0, 1080],[0, 3934, 520],[0, 0, 1]]

# flipped
# proj_bottom = torch.tensor([[-0.000403627804706846, -0.544082819271381, 0.165668907306697, 0.531281617019886],
#                             [-0.613818353794185, 0.00208996815074011, 0.0812587025493643, 0.104340549616095],
#                             [-2.41949601687988e-06, 4.01339257127901e-06, 0.000155166249656066, 0.000298115154950191]])

# normal
# proj_bottom = torch.tensor([[-0.00619327872089096,0.646890733559282,0.178814923010115,0.0925205303236082],
#                             [-0.719011631314301,0.00244867427503915,0.0951560413185957,0.122150103915150],
#                             [-3.31301811888328e-06,4.70214375197404e-06,0.000181713032159549,0.000349001099235013]
# ])

# extended long side
proj_bottom = torch.tensor([[-0.00593786102013331,	0.682233801392350,	0.171440474702767,	0.0887049230039761],
[-0.689359115872432,	0.00258246269004618,	0.0912317381188406,	0.117112553439533],
[-3.17638367162671e-06,	4.95905407620990e-06,	0.000174219058800878,	0.000334608064536045]])

distortion = [[-0.568857779226978,0.151730496415158, 0, 0, 0],[0, 0, 0, 0, 0]]

# Mean and standard deviation for normalizing input image
# IMG_NORM_MEAN = [0.485, 0.456, 0.406]
# IMG_NORM_STD = [0.229, 0.224, 0.225]

num_bone = 11

"""
Body_pose angle limit
we minius index by 1 because we exclude root pose as it is modeled as global orient
"""
max_lim = [0.] * (num_bone*3)
min_lim = [0.] * (num_bone*3)

# for i in range(num_bone):
#     max_lim[i * 3: (i + 1) * 3] = 0, 1.2, 0.02
#     min_lim[i * 3: (i + 1) * 3] = 0, -1.2, -0.02
#
# for i in range(num_bone - 4, num_bone):
#     max_lim[i * 3: (i + 1) * 3] = 0, 1.5, 0.03
#     min_lim[i * 3: (i + 1) * 3] = 0, -1.5, -0.03

for i in range(num_bone):
    max_lim[i * 3: (i + 1) * 3] = 0., 0.3, 0.
    min_lim[i * 3: (i + 1) * 3] = 0., -0.3, 0.

for i in range(num_bone - 3, num_bone):
    max_lim[i * 3: (i + 1) * 3] = 0.02, 0.4, 0.
    min_lim[i * 3: (i + 1) * 3] = 0.02, -0.4, 0.

for i in range(num_bone - 1, num_bone):
    max_lim[i * 3: (i + 1) * 3] = 0.02, 0.5, 0.
    min_lim[i * 3: (i + 1) * 3] = 0.02, -0.5, 0.

"""
Body bone length limit
"""
# max_bone = [2.4] * (num_bone)
max_bone = [2.3] * (num_bone)
min_bone = [1.0] * (num_bone)

# max_bone[0] = 2.7
# max_bone[1] = 2.7
# max_bone[2] = 2.7
# max_bone[3] = 1.7

# min_bone[0] = 0.3
# min_bone[1] = 0.5
# min_bone[2] = 0.4
# min_bone[3] = 0.4

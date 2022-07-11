import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from animal_model.fish_model import fish_model
from model.pose_optimizer import OptimizeMV
from util.Silhouette_Renderer import Silhouette_Renderer

from animal_model.MaskDatasets import Multiview_Dataset
from util import multiview
from util import multiview_utils as mutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', nargs='+', default=list(range(6520, 6700)), # (1799,1935)
                        help='Index for an example in the multiview dataset')  # 1: 10 2: 1732 [10, 150, 500, 1320, 1732]

    parser.add_argument('--mesh', type=str, default='goldfish_design_small.json', help='file of template fish')
    parser.add_argument('--outdir', type=str, default='data/output/20171216_122522', help='Folder for output images')
    parser.add_argument('--datadir', type=str, default='data/input/video_frames_20171216_122522',
                        help='Folder for input dataset')

    parser.add_argument('--eye_model', type=str, default='model/saved_models/eye_detection_model_2022-03-09',
                        help='path to mask-rcnn model for eye area detection')
    parser.add_argument('--seed', type=int, default=1, help='RNG for reproducibility')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if not os.path.exists(os.path.join(args.outdir, 'images')):
        os.mkdir(os.path.join(args.outdir, 'images'))

    # Load model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    fish = fish_model(mesh=args.mesh)
    optimizer = OptimizeMV(num_iters=200, lim_weight=200, prior_weight=30,
                           bone_weight=200, mask_weight=200, smooth_weights=[100, 100, 20],
                           device=device, mesh=args.mesh)
    silhouette_renderer = Silhouette_Renderer(device=device)

    # manual transform to debug fish model ==================
    # global_p = torch.zeros(1, 3)  # [rot_x, rot_y, rot_z]
    # # global_p[0, 0] = 9.9005e-01
    # # global_p[0, 1] = 1.4648
    # # global_p[0, 2] = 1.1059
    #
    # local_p = torch.zeros(c.num_bone, 3)
    # # local_p[2, 2] = 0.5
    # # local_p[-3, 1] = 0.5
    # local_p = local_p.view(1, c.num_bone * 3)
    #
    # bones = torch.ones(1, c.num_bone)
    # bones[0, -2] = 1
    # fish_posed = fish(global_p, local_p, bones, 1)
    #
    # vertex_posed = fish_posed['vertices']
    # with open(os.path.join(args.outdir, 'debug_out_model.obj'), 'w') as modelfile:
    #     for i in range(vertex_posed.size(1)):
    #         modelfile.write(
    #             "v {0} {1} {2}\n".format(vertex_posed[0, i, 0], vertex_posed[0, i, 1], vertex_posed[0, i, 2]))
    #
    #     for i in range(fish.faces.size(0)):
    #         modelfile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(fish.faces[i, 0] + 1, fish.faces[i, 1] + 1,
    #                                                                 fish.faces[i, 2] + 1))
    # =========================================================

    multiview_data = Multiview_Dataset(root=args.datadir)

    individual_fit_parameters = []
    # individual_fit_parameters_2 = []  # for the second fish
    sample_data = []
    # sample_data_2 = []
    prev_keypoints = None
    eye_info_list = []

    for sample_index in args.index:
        print("processing sample: {}".format(sample_index))
        sample = multiview_data[sample_index]

        if not sample['full_kpts']:
            individual_fit_parameters += [individual_fit_parameters[-4], individual_fit_parameters[-3], individual_fit_parameters[-2], individual_fit_parameters[-1]]
            sample_data.append(sample_data[-1])
            print("sample {} has missing keypoints".format(sample_index))
            continue

        frames = sample["frames"]
        img_filenames = sample["imgpaths"]
        keypoints = sample["keypoints"]
        masks = sample["masks_full"].to(device) / 255.
        bboxs = sample["bboxes"]

        keypoints_2 = torch.cat([sample['keypoints'][[0]], sample["keypoints2"][[1]]])
        masks_2 = sample["masks2"].to(device) / 255.
        bboxs_2 = sample["bboxes2"]

        prev_keypoints = keypoints.clone()

        sample_data.append([frames, img_filenames, keypoints, masks, bboxs, sample_index])
        #sample_data_2.append([frames, img_filenames, keypoints_2, masks_2, bboxs_2, sample_index])

        prev_index = sample_index - args.index[0] - 1
        if prev_index >= 0:
            init_ori = individual_fit_parameters[4 * prev_index][:, 0:3]
            init_pose = individual_fit_parameters[4 * prev_index][:, 3:]
            init_bone = individual_fit_parameters[4 * 0 + 1]
            init_s = individual_fit_parameters[4 * prev_index + 2]
            init_t = individual_fit_parameters[4 * prev_index + 3]

            # init_ori_2 = individual_fit_parameters_2[4 * prev_index][:, 0:3]
            # init_pose_2 = individual_fit_parameters_2[4 * prev_index][:, 3:]
            # init_bone_2 = individual_fit_parameters_2[4 * prev_index + 1]
            # init_s_2 = individual_fit_parameters_2[4 * prev_index + 2]
            # init_t_2 = individual_fit_parameters_2[4 * prev_index + 3]
        else:
            init_ori = None
            init_pose = None
            init_bone = None
            init_s = None
            init_t = None

            # init_ori_2 = None
            # init_pose_2 = None
            # init_bone_2 = None
            # init_s_2 = None
            # init_t_2 = None

        vertex_posed, mesh_keypoints, t, body_pose, bone, scale, losses \
            = multiview.fit_mesh(fish, optimizer, keypoints, frames, masks, silhouette_renderer, device,
                                 init_ori, init_t, init_s, init_pose, init_bone, img_filenames, sample_index, bboxs)

        # vertex_posed_2, mesh_keypoints_2, t_2, body_pose_2, bone_2, scale_2, losses_2 \
        #     = multiview.fit_mesh(fish, optimizer, keypoints_2, frames, masks_2, silhouette_renderer, device,
        #                          init_ori_2, init_t_2, init_s_2, init_pose_2, init_bone_2)

        individual_fit_parameters += [body_pose, bone, scale, t]
        #fish_posed = fish(body_pose[:,:3], body_pose[:,3:], bone, scale)

        # Save reconstruction results as images and meshes ==================
        if not os.path.exists(args.outdir):
            os.mkdir(args.outdir)

        for i in range(len(img_filenames)):
            frame = [frames[i]]
            img = plt.imread(img_filenames[i]) * 255
            img_pose, img_model = mutil.render_vertex_on_frame(img, [vertex_posed], fish, frame, [keypoints], bboxs[i])
            #img_pose, img_model = mutil.render_vertex_on_frame(img, [debug_vert], fish, frame, [keypoints], bboxs[i])

            plt.imsave(args.outdir +'/images/{}_view_{}.png'.format(sample_index, i), img_pose)
            # plt.imsave(args.outdir + '/images/{}_model_view_{}.png'.format(sample_index, i), img_model)

        # save to .obj
        with open(os.path.join(args.outdir, 'models', '{}_out_model_0.obj'.format(sample_index)), 'w') as modelfile:
            for i in range(vertex_posed.size(1)):
                modelfile.write("v {0} {1} {2}\n".format(vertex_posed[0,i,0], vertex_posed[0,i,1], vertex_posed[0,i,2]))

            for i in range(fish.faces.size(0)):
                modelfile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(fish.faces[i,0]+1, fish.faces[i,1]+1, fish.faces[i,2]+1))

        # with open(os.path.join(args.outdir, 'models', '{}_out_model_1.obj'.format(sample_index)), 'w') as modelfile:
        #     for i in range(vertex_posed_2.size(1)):
        #         modelfile.write("v {0} {1} {2}\n".format(vertex_posed_2[0,i,0], vertex_posed_2[0,i,1], vertex_posed_2[0,i,2]))
        #
        #     for i in range(fish.faces.size(0)):
        #         modelfile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(fish.faces[i,0]+1, fish.faces[i,1]+1, fish.faces[i,2]+1))

    # save posed vertices ======================
    packed_data = {"individual_fit_parameters": individual_fit_parameters,
                   "sample_data": sample_data,
                   #"individual_fit_parameters_2": individual_fit_parameters_2,
                   #"sample_data_2": sample_data_2,
                   "indices": args.index,
                   "model_file": args.mesh}

    with open(os.path.join(args.outdir, 'pose_result_{}-{}.pickle'.format(args.index[0], args.index[-1]+1)), 'wb') as pf:
        pickle.dump(packed_data, pf, protocol=pickle.HIGHEST_PROTOCOL)

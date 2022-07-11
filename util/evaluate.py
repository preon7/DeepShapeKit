import argparse
import pickle
import os
import torch
from animal_model.MaskDatasets import Multiview_Dataset
from animal_model.fish_model import fish_model
from util.Silhouette_Renderer import Silhouette_Renderer
from util.geometry import perspective_projection
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import util.multiview_utils as mutils

def kpt_distance(kpt, model_kpt, conf):
    dist = torch.sqrt(torch.square(kpt - model_kpt).sum(-1)) * conf
    return dist

def evaluate(result_indices, individual_fit_parameters, image_data, fish, device):
    silhouette_renderer = Silhouette_Renderer(device=device)
    multiview_data = Multiview_Dataset(root='../data/input/video_frames_20171216_122522')

    # silhouette rotate matrices
    Ry_90 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]).to(device)
    Rz_90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).to(device)
    proj_m, focal, center, R, T, distortion = mutils.get_cam()

    front_ious = torch.zeros(len(result_indices))
    bottom_ious = torch.zeros(len(result_indices))
    front_dists = torch.zeros((len(result_indices), 6))
    bottom_dists = torch.zeros((len(result_indices), 6))

    fish_i = 0
    #result_indices = result_indices[400:]
    pbar = trange(len(result_indices), desc="evaluating frames")
    prev_keypoints = None
    for frame in result_indices:
        sample = multiview_data[frame]
        masks = sample["masks_full"].to(device)
        keypoints = image_data[fish_i][2].to(device)

        fish_output = fish(individual_fit_parameters[4 * fish_i][:, 0:3],  # global pose
                           individual_fit_parameters[4 * fish_i][:, 3:],  # body pose
                           individual_fit_parameters[4 * fish_i + 1],  # bone length
                           individual_fit_parameters[4 * fish_i + 2])  # scale

        vertex_posed = fish_output['vertices'].to(device)

        global_t = individual_fit_parameters[4 * fish_i + 3].to(device)
        silhouette_front = silhouette_renderer(fish_output['vertices'], fish.faces.unsqueeze(0), global_t,
                                               'front') * 255.
        silhouette_bottom = silhouette_renderer(fish_output['vertices'], fish.faces.unsqueeze(0), global_t,
                                                'bottom') * 255.

        # front IoU
        front_bin_mask = masks[0] > 200
        front_bin_silh = silhouette_front[0] > 0

        plt.imsave('../data/output/multiview_demo/front_bin_mask.png', front_bin_mask.detach().cpu().numpy())
        plt.imsave('../data/output/multiview_demo/front_bin_silh.png', front_bin_silh.detach().cpu().numpy())

        intersection = torch.logical_and(front_bin_silh, front_bin_mask)
        union = torch.logical_or(front_bin_silh, front_bin_mask)
        front_ious[fish_i] = intersection.sum() / union.sum()

        # bottom IoU
        bottom_bin_mask = masks[1] > 200
        bottom_bin_silh = silhouette_bottom[0] > 0

        plt.imsave('../data/output/multiview_demo/bottom_bin_mask.png', bottom_bin_mask.detach().cpu().numpy())
        plt.imsave('../data/output/multiview_demo/bottom_bin_silh.png', bottom_bin_silh.detach().cpu().numpy())

        intersection = torch.logical_and(bottom_bin_silh, bottom_bin_mask)
        union = torch.logical_or(bottom_bin_silh, bottom_bin_mask)
        bottom_ious[fish_i] = intersection.sum() / union.sum()

        # keypoint error
        global_t = individual_fit_parameters[4 * fish_i + 3].to(device)
        model_keypoints = fish_output['keypoints'].to(device) + global_t.repeat(1, 1, 1)
        model_keypoints = model_keypoints.repeat([2, 1, 1])

        projected_keypoints = perspective_projection(model_keypoints, proj_m)
        kpt_dist = kpt_distance(keypoints[:, :, :2], projected_keypoints, keypoints[:, :, -1])
        front_dists[fish_i, :] = kpt_dist[0]
        bottom_dists[fish_i, :] = kpt_dist[1]

        fish_i += 1
        pbar.update()

    print("average front IoU: {}".format(front_ious.mean().item()))
    print("IoU >= 0.5: {}".format((front_ious >= 0.5).sum().item()))
    print("IoU >= 0.75: {}".format((front_ious >= 0.75).sum().item()))

    print("average bottom IoU: {}".format(bottom_ious.mean().item()))
    print("IoU >= 0.5: {}".format((bottom_ious >= 0.5).sum().item()))
    print("IoU >= 0.75: {}".format((bottom_ious >= 0.75).sum().item()))

    print("average front kpt distance")
    print(front_dists.mean(0))
    front_mean = front_dists.mean(0)
    front_mean[3] = 0
    print(front_mean.sum() / (front_mean.size(0) - 1))

    print("average bottom kpt distance")
    print(bottom_dists.mean(0))
    bottom_mean = bottom_dists.mean(0)
    bottom_mean[-1] = 0
    #print(bottom_mean.mean())
    print(bottom_mean.sum() / (bottom_mean.size(0) - 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mesh', type=str, default='goldfish_design_small.json', help='file of template fish')
    parser.add_argument('--in_file', type=str, default='../data/output/multiview_demo/pose_pickle/pose_result_occ.pickle', help='pickle file location')
    parser.add_argument('--datadir', type=str, default='../data/input/video_frames_21-01-2022',
                        help='Folder for input dataset')
    parser.add_argument('--seed', type=int, default=1, help='RNG for reproducibility')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pose_dic = pickle.load(open(args.in_file, 'rb'))
    fish = fish_model(mesh=args.mesh)

    # load result from pickle
    result_indices = pose_dic["indices"]
    individual_fit_parameters = pose_dic["individual_fit_parameters"]
    image_data = pose_dic['sample_data']

    # load data
    multiview_data = Multiview_Dataset(root=args.datadir)

    evaluate(result_indices, individual_fit_parameters, image_data, device, fish)
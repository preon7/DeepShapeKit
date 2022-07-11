import argparse
import pickle
import torch
from animal_model.MaskDatasets import Multiview_Dataset
from animal_model.fish_model import fish_model
from util.Silhouette_Renderer import Silhouette_Renderer
from tqdm import tqdm, trange
import util.multiview_utils as mutils

def kpt_distance(kpt, model_kpt, conf):
    dist = torch.sqrt(torch.square(kpt - model_kpt).sum(-1)) * conf
    return dist

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

    print(f"body scale: {pose_dic['individual_fit_parameters'][2]}")

    silhouette_renderer = Silhouette_Renderer(380, torch.tensor([[-0.3, 0, 0]]), device)

    # load result from pickle
    result_indices = pose_dic["indices"]
    individual_fit_parameters = pose_dic["individual_fit_parameters"]
    image_data = pose_dic['sample_data']

    # load data
    multiview_data = Multiview_Dataset(root=args.datadir)

    # silhouette rotate matrices
    Ry_90 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]).to(device)
    Rz_90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).to(device)
    proj_m, focal, center, R, T, distortion = mutils.get_cam()

    vert_dists = torch.zeros(len(result_indices) - 1)
    pose_dists = torch.zeros(len(result_indices) - 1)

    fish_i = 0
    pbar = trange(len(result_indices), desc="evaluating frames")
    prev_vertices = None
    for frame in result_indices:

        masks = image_data[fish_i][3] * 255
        keypoints = image_data[fish_i][2].to(device)
        poses = individual_fit_parameters[4 * fish_i]

        fish_output = fish(individual_fit_parameters[4 * fish_i][:, 0:3],  # global pose
                           individual_fit_parameters[4 * fish_i][:, 3:],  # body pose
                           individual_fit_parameters[4 * fish_i + 1],  # bone length
                           individual_fit_parameters[4 * fish_i + 2])  # scale

        vertex_posed = fish_output['vertices'].to(device)
        if fish_i != 0:
            vert_dists[fish_i - 1] = torch.sqrt(torch.square(vertex_posed[0] - prev_vertices[0]).sum(-1)).mean() / pose_dic['individual_fit_parameters'][4 * fish_i + 2]
            prev_poses = individual_fit_parameters[4 * (fish_i - 1)]
            pose_dists[fish_i - 1] = torch.abs(poses - prev_poses).sum()

        prev_vertices = vertex_posed.clone()

        fish_i += 1
        pbar.update()

    print('average scaled vertex smoothness: {}'.format(vert_dists.mean().item()))
    print('average pose smoothness: {}'.format(pose_dists.mean().item()))
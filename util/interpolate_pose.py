import csv
import os

import numpy as np
import torch
import pickle
import argparse
from util import constants as c
from evaluate import evaluate

from animal_model.fish_model import fish_model
from util import constants
from tqdm import tqdm, trange
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1):
        super().__init__()
        self.bilstm = nn.LSTM(in_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.bilstm2 = nn.LSTM(2 * hidden_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, in_size)

    def __call__(self, sequence):
        lstm_out, (hn, cn) = self.bilstm(sequence)
        #lstm_out, (hn, cn) = self.bilstm2(lstm_out)
        return self.fc(lstm_out)

def pred_loss(sequence, predicted, seq_weight, lim_weight, global_weight=0.01):
    L2 = nn.MSELoss()
    loss = L2(sequence, predicted)
    for i in range(predicted.size(0)):
        if i == 0:
            continue
        # loss += seq_weight * L2(predicted[i,0,3:], predicted[i - 1,0,3:])
        # loss += seq_weight * L2(predicted[i, 0, 0:3], predicted[i - 1, 0, 0:3])
        loss += seq_weight * L2(predicted[i, 0, :], predicted[i - 1, 0, :])

    # Joint angle limit loss
    # max_lim = torch.tensor(constants.max_lim).repeat(1, 1).to(device)
    # min_lim = torch.tensor(constants.min_lim).repeat(1, 1).to(device)
    #
    # for i in range(predicted.size(0)):
    #     lim_loss = (predicted[i,0,3:] - max_lim).clamp(0, float("Inf")) + (min_lim - predicted[i,0,3:]).clamp(0, float("Inf"))
    #     loss += lim_weight * lim_loss.sum()

    return loss

def denoise_sequence(args, sequence, device):
    model = BiLSTM(sequence.size(2), 250).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)

    with trange(args.epoches) as tepoch:
        for e in tepoch:

            optimizer.zero_grad()

            predicted = model(sequence)
            loss = pred_loss(sequence, predicted, 0.05, 0.0000)

            loss.backward()
            optimizer.step()
            tepoch.set_description(f"Epoch {e}, loss: {loss.item()}")

            tepoch.set_postfix(loss=loss.item())

    return predicted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--index', nargs='+', default=list(range(0, 200)), #265
    #                     help='Index for an example in the multiview dataset')
    parser.add_argument('--mesh', type=str, default='goldfish_design_small.json', help='file of template fish')
    parser.add_argument('--dir', type=str, default='../data/output/20171216_122522', help='Folder contains pickle file')
    parser.add_argument('-f', '--file', type=str, default='pose_pickle/pose_result_0-159_(1).pickle', help='pickle file name')
    # parser.add_argument('-f', '--file', type=str, default='pose_pickle/ablation_simple_400-600.pickle',
    #                     help='pickle file name')
    parser.add_argument('-s', '--interpolate_size', type=int, default=1, help='number of splits between frames')

    parser.add_argument('--denoise', type=bool, default=True, help='use bi-lstm to denoise pose sequence')
    parser.add_argument('-e', '--epoches', type=int, default=150, help='number of epoches for denoising')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, )


    args = parser.parse_args()
    pose_dic = pickle.load(open(os.path.join(args.dir, args.file), 'rb'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fish = fish_model(mesh=args.mesh, device=device)

    # calculate distance between two models
    # distance_file = open(args.dir + '/vector_A2B.csv', 'w')
    # csv_writer = csv.writer(distance_file, delimiter=',',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # csv_writer.writerow(['frame', 'x', 'y', 'z'])
    #
    # dist_list = []
    # for i in range(len(args.index)):
    #     dist_vec = pose_dic['individual_fit_parameters_2'][4 * i + 3] - \
    #                pose_dic['individual_fit_parameters'][4 * i + 3]
    #     dist_list.append(dist_vec)

        # csv_writer.writerow([i, dist_vec[0].item(), dist_vec[1].item(), dist_vec[2].item()])

    indices = pose_dic["indices"][:]

    for model_i in range(2):
        if model_i == 0:
            #continue
            param_id = 'individual_fit_parameters'
            data_id = 'sample_data'
            model_name = 'A'
        elif model_i == 1:
            continue  # for one model case
            param_id = 'individual_fit_parameters_2'
            data_id = 'sample_data_2'
            model_name = 'B'

        bone_length = pose_dic[param_id][1].to(device)  # not changing across frames
        scale = pose_dic[param_id][2].to(device)
        image_data = pose_dic[data_id]

        sequence = []
        for i in range(10):
            sequence.append(pose_dic[param_id][0].unsqueeze(0))

        for i in range(len(indices)):
            sequence.append(pose_dic[param_id][4 * i].unsqueeze(0))

        for i in range(10):
            sequence.append(pose_dic[param_id][-4].unsqueeze(0))

        #sequence = torch.cat(sequence).to(device)[10:-10]
        sequence = torch.cat(sequence).to(device)
        sequence = denoise_sequence(args, sequence, device)[10:-10]

        # for i in args.index:
        #     if i not in pose_dic['indices']:
        #         raise Exception("input index {} not stored in file".format(i))

        # smoothness after optimization
        vert_dists = torch.zeros(len(indices) - 1)
        pose_dists = torch.zeros(len(indices) - 1)

        smoothed_fit_parameters = []

        # output pose to csv
        start = image_data[0][-1]
        end = image_data[-1][-1]
        pose_csv = open(os.path.join(args.dir, f'pose_data_{start}-{end}_smooth.csv'), 'w')

        prev_vert = None
        pbar = trange(len(indices) - 1, desc="outputting frames")
        for i in range(len(indices)):
            for bone in range(12):
                pose_csv.write(str(sequence[i][0][bone * 3:bone * 3 + 3].tolist()) + '|')
            pose_csv.write('\n')

            smoothed_fit_parameters += [sequence[i], bone_length, scale, pose_dic[param_id][4 * i+3]]
            fish_output = fish(sequence[i][:, 0:3],  # global pose
                               sequence[i][:, 3:],  # body pose
                               bone_length,  # bone length
                               scale)  # scale

            vertex_posed = fish_output['vertices'].to(device)

            # save to .obj
            # with open(os.path.join(args.dir, 'interp_models', '{}_out_model_{}.obj'.format(i, model_name)),
            #           'w') as modelfile:
            #     for j in range(vertex_posed.size(1)):
            #         modelfile.write(
            #             "v {0} {1} {2}\n".format(vertex_posed[0, j, 0], vertex_posed[0, j, 1], vertex_posed[0, j, 2]))
            #
            #     for j in range(fish.faces.size(0)):
            #         modelfile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(fish.faces[j, 0] + 1, fish.faces[j, 1] + 1,
            #                                                                 fish.faces[j, 2] + 1))

            if i != 0:
                vert_dists[i - 1] = torch.sqrt(torch.square(vertex_posed[0] - prev_vert[0]).sum(-1)).mean() / pose_dic['individual_fit_parameters'][4 * i + 2]
                prev_poses = sequence[i-1]
                pose_dists[i - 1] = torch.abs(sequence[i] - prev_poses).sum()

            prev_vert = vertex_posed.clone()
            pbar.update(1)

        pose_csv.close()
        print('average scaled vertex smoothness: {}'.format(vert_dists.mean().item()))
        print('average pose smoothness: {}'.format(pose_dists.mean().item()))

        evaluate(indices, smoothed_fit_parameters, image_data, fish, device)

        # interpolate
        # model_count = 0
        # pbar = trange(len(pose_dic["indices"]) - 1, desc="interpolating frames")
        # for count, index in enumerate(pose_dic["indices"]): # enumerate(pose_dic['indices']):
        #     if count == 0:
        #         continue
        #
        #     pbar.set_description('interpolating frame {} and {}'.format(count - 1, count))
        #     # pose_array_0 = pose_dic['individual_fit_parameters'][4 * (count - 1)].numpy()
        #     # pose_array_1 = pose_dic['individual_fit_parameters'][4 * count].numpy()
        #     pose_array_0 = sequence[count - 1].cpu().detach().numpy()
        #     pose_array_1 = sequence[count].cpu().detach().numpy()
        #
        #     xs = [0, args.interpolate_size]
        #     samples = list(range(args.interpolate_size))
        #
        #     results = np.zeros((args.interpolate_size, pose_array_0.shape[1]))
        #     for i in range(pose_array_0.shape[1]):
        #         sampled_points = np.interp(samples, xs, [pose_array_0[0,i], pose_array_1[0,i]])
        #         results[:,i] = sampled_points
        #
        #     results = torch.from_numpy(results)
        #
        #     # dist_array_0 = dist_list[count - 1].cpu().detach().numpy()
        #     # dist_array_1 = dist_list[count].cpu().detach().numpy()
        #     #
        #     # dist_interp = np.zeros((args.interpolate_size, dist_array_0.shape[0]))
        #     # for i in range(dist_array_0.shape[0]):
        #     #     sampled_points = np.interp(samples, xs, [dist_array_0[i], dist_array_1[i]])
        #     #     dist_interp[:, i] = sampled_points
        #     #
        #     # dist_interp = torch.from_numpy(dist_interp)
        #
        #     for i in range(args.interpolate_size):
        #         # if model_i == 0:
        #         #     csv_writer.writerow([model_count, dist_interp[i,0].item(), dist_interp[i,1].item(), dist_interp[i,2].item()])
        #
        #         global_pose = results[[i]][:, 0:3].to(device)
        #         local_pose = results[[i]][:, 3:].to(device)
        #
        #         fish_output = fish(global_pose,  # global pose
        #                            local_pose,  # body pose
        #                            bone_length,  # bone length
        #                            scale)  # scale
        #
        #         vertex_posed = fish_output['vertices']
        #         # save to .obj
        #         with open(os.path.join(args.dir, 'interp_models', '{}_out_model_{}.obj'.format(model_count, model_name)), 'w') as modelfile:
        #             for i in range(vertex_posed.size(1)):
        #                 modelfile.write("v {0} {1} {2}\n".format(vertex_posed[0,i,0], vertex_posed[0,i,1], vertex_posed[0,i,2]))
        #
        #             for i in range(fish.faces.size(0)):
        #                 modelfile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(fish.faces[i,0]+1, fish.faces[i,1]+1, fish.faces[i,2]+1))
        #
        #             model_count += 1
        #
        #     pbar.update(1)

    # distance_file.close()
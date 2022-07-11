import csv
import os
import cv2
import numpy as np
import skimage
import torch
import pickle
import json

from PIL import Image
from util import mrcnn_utils as utils
# from util import constants

class Multiview_Dataset(torch.utils.data.Dataset):
    def __init__(self, root='data/detection_1/frames',
                 front_kpts_file='full_sizeDLC_resnet50_master2021demo_frontJun2shuffle1_30000_full.pickle',
                 bottom_kpts_file='full_sizeDLC_resnet50_master2021demo_bottomJun1shuffle1_15000_full.pickle'):
        self.root = root

        self.detections_front = pickle.load(open(os.path.join(root, "sample-front", "dlc_results",front_kpts_file), "rb"))
        self.detections_bottom = pickle.load(open(os.path.join(root, "sample-bottom", "dlc_results",bottom_kpts_file), "rb"))

        self.front_index = {}
        with open(os.path.join(root, "sample-front", "files.csv")) as fif:
            reader = csv.DictReader(fif)
            for row in reader:
                self.front_index[row['frame']] = row

        self.bottom_index = {}
        with open(os.path.join(root, "sample-bottom", "files.csv")) as bif:
            reader = csv.DictReader(bif)
            for row in reader:
                self.bottom_index[row['frame']] = row

        self.mask_front_index = {}
        with open(os.path.join(root, "sample-front", "files_crop.csv")) as fif:
            reader = csv.DictReader(fif, quotechar=' ')
            for row in reader:
                if row['category'] == 'mask' and row['sub_index'] == '0':
                    self.mask_front_index[row['frame']] = [row]
                elif row['category'] == 'mask' and row['sub_index'] == '1':
                    self.mask_front_index[row['frame']].append(row)

        self.mask_bottom_index = {}
        with open(os.path.join(root, "sample-bottom", "files_crop.csv")) as bif:
            reader = csv.DictReader(bif, quotechar=' ')
            for row in reader:
                if row['category'] == 'mask' and row['sub_index'] == '0':
                    if row['frame'] not in self.mask_bottom_index.keys():
                        self.mask_bottom_index[row['frame']] = [row]
                    else:
                        self.mask_bottom_index[row['frame']].append(row)
                elif row['category'] == 'mask' and row['sub_index'] == '1':
                    if row['frame'] not in self.mask_bottom_index.keys():
                        self.mask_bottom_index[row['frame']] = [row]
                    else:
                        self.mask_bottom_index[row['frame']].append(row)

        self.front_match = {}
        with open(os.path.join(root, "sample-front", "frame2video_1.csv")) as fif:
            reader = csv.DictReader(fif)
            for row in reader:
                self.front_match[row['new_frame']] = row['origin_frame']

        self.bottom_match = {}
        with open(os.path.join(root, "sample-bottom", "frame2video_1.csv")) as bif:
            reader = csv.DictReader(bif)
            for row in reader:
                self.bottom_match[row['new_frame']] = row['origin_frame']

    def __getitem__(self, index):
        path_front = self.front_index[self.front_match[str(index)]]['file_loc']
        path_bottom = self.bottom_index[self.bottom_match[str(index)]]['file_loc']

        front_bboxs = [row['bbox'] for row in self.mask_front_index[self.front_match[str(index)]]]
        bbox_front = []
        for candidate_box in front_bboxs:
            bbox_list = []
            for x in candidate_box[1:-1].split(','):
                try:
                    bbox_list.append(int(x))
                except:
                    pass
            if len(bbox_list) != 0:
                bbox_front.append(bbox_list)

        front_individual_index = 0
        if len(bbox_front) == 2:
            if bbox_front[0][0] < bbox_front[1][0]:
                bbox_front2 = bbox_front[1]
                bbox_front = bbox_front[0]
                row_front = self.mask_front_index[self.front_match[str(index)]][0]
                row_front2 = self.mask_front_index[self.front_match[str(index)]][1]
            else:
                front_individual_index = 1
                bbox_front2 = bbox_front[0]
                bbox_front = bbox_front[1]
                row_front = self.mask_front_index[self.front_match[str(index)]][1]
                row_front2 = self.mask_front_index[self.front_match[str(index)]][0]
        else:  # len == 1
            #return {'full_kpts': False}
            bbox_front = bbox_front[0]
            bbox_front2 = bbox_front
            row_front = self.mask_front_index[self.front_match[str(index)]][0]
            row_front2 = row_front

        bottom_bboxs = [row['bbox'] for row in self.mask_bottom_index[self.bottom_match[str(index)]]]
        bbox_bottom = []

        for candidate_box in bottom_bboxs:
            bbox_list = []
            for x in candidate_box[1:-1].split(','):
                try:
                    bbox_list.append(int(x))
                except:
                    pass
            if len(bbox_list) != 0:
                bbox_bottom.append(bbox_list)

        bottom_individual_index = 0
        if len(bbox_bottom) == 2:
            if bbox_bottom[0][0] < bbox_bottom[1][0]:
                bbox_bottom2 = bbox_bottom[1]
                bbox_bottom = bbox_bottom[0]
                row_bottom = self.mask_bottom_index[self.bottom_match[str(index)]][0]
                row_bottom2 = self.mask_bottom_index[self.bottom_match[str(index)]][1]
            else:
                bottom_individual_index = 1
                bbox_bottom2 = bbox_bottom[0]
                bbox_bottom = bbox_bottom[1]
                row_bottom = self.mask_bottom_index[self.bottom_match[str(index)]][1]
                row_bottom2 = self.mask_bottom_index[self.bottom_match[str(index)]][0]
        else:  # len == 1
            return {'full_kpts': False}
            bbox_bottom2 = None
            bbox_bottom = bbox_bottom[0]
            row_bottom = self.mask_bottom_index[self.bottom_match[str(index)]][0]
            row_bottom2 = None

        # bbox_front = self.mask_front_index[self.front_match[str(index)]]['bbox']
        # bbox_front_list = []
        # for x in bbox_front[1:-1].split(' '):
        #     try:
        #         bbox_front_list.append(int(x))
        #     except:
        #         pass
        # if len(bbox_front_list) != 0:
        #     bbox_front = bbox_front_list  # [w_low, h_low, w_up, h_up]
        # else:
        #     raise Exception('front bbox empty')
        #
        # bbox_bottom = self.mask_bottom_index[self.bottom_match[str(index)]]['bbox']
        # # bbox_bottom = [int(x) for x in bbox_bottom[1:-1].split(' ')]
        # bbox_bottom_list = []
        # for x in bbox_bottom[1:-1].split(' '):
        #     try:
        #         bbox_bottom_list.append(int(x))
        #     except:
        #         pass
        # if len(bbox_bottom_list) != 0:
        #     bbox_bottom = bbox_bottom_list
        # else:
        #     raise Exception('bottom bbox empty')

        # get keypoints from pickle files
        item_key = 'frame' + '0' * (5 - len(str(index))) + str(index)

        front_confdences = self.detections_front[item_key]['confidence']
        front_kpts = self.detections_front[item_key]['coordinates'][0]
        kpt_front = []
        kpt_front2 = []
        for i in range(len(front_confdences)):
            if i == 5:  # skip eye keypoint
                continue
            #kpt_id = np.argmax(front_confdences[i])
            kpt_ids = front_confdences[i][:, 0].argsort()[-2:]  # two most confident kpt
            if len(kpt_ids) < 2:
                kpt_id = kpt_ids[0]
            else:
                #kpt_id = kpt_ids[1]  # np.argmin(front_kpts[i][:2, 0])
            # kpt_front.append(np.append(front_kpts[i][kpt_id],front_confdences[i][kpt_id]))
            # kpt_id = kpt_ids[0]  # np.argmax(front_kpts[i][:2, 0])
            # kpt_front2.append(np.append(front_kpts[i][kpt_id], front_confdences[i][kpt_id]))
                # check if kpt in bounding box
                kpt_id = kpt_ids[0]  # np.argmax(bottom_kpts[i][:2, 0])
                kpt_id_2 = kpt_ids[1]
                w1 = front_kpts[i][kpt_id, 0]
                w2 = front_kpts[i][kpt_id_2, 0]
                h1 = front_kpts[i][kpt_id, 1]
                h2 = front_kpts[i][kpt_id_2, 1]

                # if bbox_front[0] - 5 < w1 < bbox_front[2] + 5 and bbox_front[1] - 5 < h1 < bbox_front[3] + 5:
                #     kpt_front.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
                #     kpt_front2.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))
                # else:
                #     kpt_front2.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
                #     kpt_front.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))

                if i in [0, 1]:
                    bound_w = [bbox_front[0] - 5, (bbox_front[0] + bbox_front[2]) / 2]
                    bound_h = [bbox_front[1] - 5, bbox_front[3] + 5]
                elif i in [2, 3, 6]:
                    bound_w = [(bbox_front[0] + bbox_front[2]) / 2, bbox_front[2] + 5]
                    bound_h = [bbox_front[1] - 5, bbox_front[3] + 5]
                else:
                    bound_w = [bbox_front[0] - 5, bbox_front[2] + 5]
                    bound_h = [bbox_front[1] - 5, bbox_front[3] + 5]

                if bound_w[0] < w1 < bound_w[1] and bound_h[0] < h1 < bound_h[1]:
                    kpt_front.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
                    kpt_front2.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))
                elif bound_w[0] < w2 < bound_w[1] and bound_h[0] < h2 < bound_h[1]:
                    kpt_front2.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
                    kpt_front.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))
                else:
                    dist1 = (w1 - ((bbox_front[0] + bbox_front[2]) / 2)) ** 2 + (
                            h1 - ((bbox_front[1] + bbox_front[3]) / 2)) ** 2
                    dist2 = (w2 - ((bbox_front[0] + bbox_front[2]) / 2)) ** 2 + (
                            h2 - ((bbox_front[1] + bbox_front[3]) / 2)) ** 2
                    if dist1 < dist2:
                        kpt_front.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
                        kpt_front2.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))
                    else:
                        kpt_front2.append(np.array([w1, h1, front_confdences[i][kpt_id, 0]]))
                        kpt_front.append(np.array([w2, h2, front_confdences[i][kpt_id_2, 0]]))

            # if (len(front_kpts[i]) > 1):
            #     kpt_front.append(np.append(front_kpts[i][front_individual_index], front_confdences[i][front_individual_index]))
            # else:
            #     kpt_front.append(
            #         np.append(front_kpts[i][0], front_confdences[i][0]))

        # front_eyes = []
        # kpt_front_new = []
        # kpt_front2_new = []
        # for i in range(len(kpt_front)):
        #     if i == 5:
        #         front_eyes.append(kpt_front[i])
        #         front_eyes.append(kpt_front2[i])
        #     else:
        #         kpt_front_new.append(kpt_front[i])
        #         kpt_front2_new.append(kpt_front2[i])
        #
        # kpt_front = kpt_front_new
        # kpt_front2 = kpt_front2_new

        if index == 1:
            print('debug')
        bottom_confdences = self.detections_bottom[item_key]['confidence']
        if len(self.detections_bottom[item_key]['coordinates']) > 1:
            print('debug')
        bottom_kpts = self.detections_bottom[item_key]['coordinates'][0]
        kpt_bottom = []
        kpt_bottom2 = []
        for i in range(len(bottom_confdences)):
            if bottom_confdences[i].shape[0] == 0:
                return {'full_kpts': False}
            if bottom_confdences[i].shape[0] < 3:
                kpt_ids = [0,1]
            else:
                kpt_ids = bottom_confdences[i][:, 0].argsort()[-2:]  # two most confident kpt
            #kpt_id = np.argmax(bottom_confdences[i])

            # check if kpt in bounding box
            kpt_id = kpt_ids[0]  # np.argmax(bottom_kpts[i][:2, 0])
            kpt_id_2 = kpt_ids[1]
            w1 = 2048 - bottom_kpts[i][kpt_id, 0]
            w2 = 2048 - bottom_kpts[i][kpt_id_2,0]
            h1 = bottom_kpts[i][kpt_id, 1]
            h2 = bottom_kpts[i][kpt_id_2, 1]

            k1_in = False
            k2_in = False
            if bbox_bottom[0] < w1 < bbox_bottom[2] and bbox_bottom[1] < h1 < bbox_bottom[3]:
                # kpt_bottom.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id,0]]))
                # kpt_bottom2.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2,0]]))
                k1_in = True
            if bbox_bottom[0] < w2 < bbox_bottom[2] and bbox_bottom[1] < h2 < bbox_bottom[3] :
                # kpt_bottom2.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
                # kpt_bottom.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))
                k2_in = True

            if k1_in and not k2_in:
                kpt_bottom.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
                kpt_bottom2.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))
            elif k2_in and not k1_in:
                kpt_bottom2.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
                kpt_bottom.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))
            else:
                dist1 = (w1 - ((bbox_bottom[0] + bbox_bottom[2]) / 2)) ** 2 + (
                            h1 - ((bbox_bottom[1] + bbox_bottom[3]) / 2)) ** 2
                dist2 = (w2 - ((bbox_bottom[0] + bbox_bottom[2]) / 2)) ** 2 + (
                            h2 - ((bbox_bottom[1] + bbox_bottom[3]) / 2)) ** 2
                if dist1 < dist2:
                    kpt_bottom.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
                    kpt_bottom2.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))
                else:
                    kpt_bottom2.append(np.array([2048 - w1, h1, bottom_confdences[i][kpt_id, 0]]))
                    kpt_bottom.append(np.array([2048 - w2, h2, bottom_confdences[i][kpt_id_2, 0]]))

            # if (len(bottom_kpts[i]) > 1):
            #     kpt_bottom.append(np.append(bottom_kpts[i][bottom_individual_index], bottom_confdences[i][bottom_individual_index]))
            # else:
            #     kpt_bottom.append(
            #         np.append(bottom_kpts[i][0], bottom_confdences[i][0]))

        kpt_front = torch.tensor(kpt_front)
        kpt_bottom = torch.tensor(kpt_bottom)
        kpt_front2 = torch.tensor(kpt_front2)
        kpt_bottom2 = torch.tensor(kpt_bottom2)

        # kpt_front = torch.cat([torch.tensor(self.detections_front[item_key]['coordinates']).squeeze(),
        #                        torch.tensor(self.detections_front[item_key]['confidence']).squeeze(1)], -1)
        # kpt_bottom = torch.cat([torch.tensor(self.detections_bottom[item_key]['coordinates']).squeeze(),
        #                        torch.tensor(self.detections_bottom[item_key]['confidence']).squeeze(1)], -1)

        width = bbox_front[2] - bbox_front[0]
        height = bbox_front[3] - bbox_front[1]
        margin_front = int((width - height) / 2)
        #kpt_front[:,0] = (width / 368.0) * kpt_front[:,0] + bbox_front[0]
        #kpt_front[:,1] = (width / 368.0) * kpt_front[:,1] + bbox_front[1] - 0.5 * (width - height)

        # undistort keypoints
        # kpt_image = np.zeros((constants.in_size[0], constants.in_size[1], kpt_front.shape[0]))
        # for i in range(kpt_front.shape[0]):
        #     kpt_image[kpt_front[i,0].int(), kpt_front[i,1].int(), i] = 5
        #     kpt_image[kpt_front[i, 0].int()-1, kpt_front[i, 1].int(), i] = 5
        #     kpt_image[kpt_front[i, 0].int()+1, kpt_front[i, 1].int(), i] = 5
        #     kpt_image[kpt_front[i, 0].int(), kpt_front[i, 1].int()-1, i] = 5
        #     kpt_image[kpt_front[i, 0].int(), kpt_front[i, 1].int()+1, i] = 5
        #
        # kpt_image = kpt_image.transpose()
        # intrinsic_mat = np.array(constants.intrinsic_mat)
        # distortion = np.array(constants.distortion[0])
        # h, w = kpt_image.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_mat, distortion, (w, h), 1, (w, h))
        #
        # mapx, mapy = cv2.initUndistortRectifyMap(intrinsic_mat, distortion, None, newcameramtx, (w, h), 5)
        # for i in range(kpt_front.shape[0]):
        #     undistort_kpt = cv2.undistort(kpt_image[i], intrinsic_mat, distortion, None, newcameramtx)

        width = bbox_bottom[2] - bbox_bottom[0]
        height = bbox_bottom[3] - bbox_bottom[1]
        margin_bottom = int((width - height) / 2)
        kpt_bottom[:, 0] = 2048. - kpt_bottom[:,0]  # for flipped detection
        kpt_bottom2[:, 0] = 2048. - kpt_bottom2[:, 0]
        # kpt_bottom[:,0] = (width / 368.0) * kpt_bottom[:,0] + bbox_bottom[0]
        # kpt_bottom[:,1] = (width / 368.0) * kpt_bottom[:,1] + bbox_bottom[1] - 0.5 * (width - height)

        kpt_bottom = torch.cat([kpt_bottom, torch.zeros([1,3])])
        kpt_bottom2 = torch.cat([kpt_bottom2, torch.zeros([1,3])])

        # img_size = 368
        # front_mask = cv2.resize(cv2.imread(os.path.join(self.root, 'front_mask', self.front_masks[index])), (img_size, img_size))
        # bottom_mask = cv2.resize(cv2.imread(os.path.join(self.root, 'bottom_mask', self.bottom_masks[index])), (img_size, img_size))

        # boxes = []
        # masks = [front_mask, bottom_mask]
        # for i in range(2):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        front_mask_crop = cv2.imread(os.path.join(self.root, row_front['file_loc']), cv2.IMREAD_GRAYSCALE)
        bottom_mask_crop = cv2.imread(os.path.join(self.root, row_bottom['file_loc']), cv2.IMREAD_GRAYSCALE)

        front_mask_crop = cv2.resize(front_mask_crop, (380, 380))
        bottom_mask_crop = cv2.resize(bottom_mask_crop, (380, 380))

        front_mask_crop2 = cv2.imread(os.path.join(self.root, row_front2['file_loc']), cv2.IMREAD_GRAYSCALE)
        bottom_mask_crop2 = cv2.imread(os.path.join(self.root, row_bottom2['file_loc']), cv2.IMREAD_GRAYSCALE)

        front_mask_crop2 = cv2.resize(front_mask_crop2, (380, 380))
        bottom_mask_crop2 = cv2.resize(bottom_mask_crop2, (380, 380))

        full_front_mask_loc = row_front['file_loc'].split('/')
        full_front_mask_loc[-2] = 'mask_full'
        front_mask_full = cv2.imread(os.path.join(self.root, '/'.join(full_front_mask_loc)), cv2.IMREAD_GRAYSCALE)

        full_bottom_mask_loc = row_bottom['file_loc'].split('/')
        full_bottom_mask_loc[-2] = 'mask_full'
        bottom_mask_full = cv2.imread(os.path.join(self.root, '/'.join(full_bottom_mask_loc)), cv2.IMREAD_GRAYSCALE)

        full_front_mask_loc2 = row_front2['file_loc'].split('/')
        full_front_mask_loc2[-2] = 'mask_full'
        front_mask_full2 = cv2.imread(os.path.join(self.root, '/'.join(full_front_mask_loc2)), cv2.IMREAD_GRAYSCALE)

        full_bottom_mask_loc2 = row_bottom2['file_loc'].split('/')
        full_bottom_mask_loc2[-2] = 'mask_full'
        bottom_mask_full2 = cv2.imread(os.path.join(self.root, '/'.join(full_bottom_mask_loc2)), cv2.IMREAD_GRAYSCALE)

        # front_mask = np.zeros((1040, 2048))
        # bottom_mask = np.zeros((1040, 2048))
        #
        # front_mask[bbox_front[1]:bbox_front[3], bbox_front[0]:bbox_front[2]] = front_mask_crop[margin_front:-margin_front,:,-1]
        # bottom_mask[bbox_bottom[1]:bbox_bottom[3], bbox_bottom[0]:bbox_bottom[2]] = bottom_mask_crop[margin_bottom:-margin_bottom,:,-1]

        data = {'imgpaths': [os.path.join(self.root, path_front), os.path.join(self.root, path_bottom)],
                'frames': [0, 1],
                'keypoints': torch.stack([kpt_front, kpt_bottom]),
                'keypoints2': torch.stack([kpt_front2, kpt_bottom2]),
                'masks': torch.stack([torch.from_numpy(front_mask_crop), torch.from_numpy(bottom_mask_crop)]),
                'masks2': torch.stack([torch.from_numpy(front_mask_crop2), torch.from_numpy(bottom_mask_crop2)]),
                'masks_full': torch.stack([torch.from_numpy(front_mask_full), torch.from_numpy(bottom_mask_full)]),
                'masks_full2': torch.stack([torch.from_numpy(front_mask_full2), torch.from_numpy(bottom_mask_full2)]),
                'bboxes': torch.tensor([bbox_front, bbox_bottom]),
                'bboxes2': torch.tensor([bbox_front2, bbox_bottom2]),
                'full_kpts': True,
                #'detected_eyes': torch.tensor(front_eyes)
                }

        return data

    def __len__(self):
        return len(self.front_match.keys())

class CocoDetection(torch.utils.data.Dataset):
    """ dataset in coco format

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.ids)


class DeepFishDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images/valid"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks/valid"))))

    def __getitem__(self, idx):
        # load images and masks
        img_name = self.imgs[idx]
        mask_name = self.masks[idx]
        if img_name.split('.')[0] != mask_name.split('.')[0]:
            print(img_name)
            print(mask_name)
            raise Exception('image: ' + img_name + 'image and mask do not match' + mask_name)

        img_path = os.path.join(self.root, "images/valid", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks/valid", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class UniKNDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_name = self.imgs[idx]
        mask_name = self.masks[idx]
        if img_name.split('.')[0] != mask_name.split('.')[0]:
            raise Exception('image and mask do not match')

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path).convert("RGB")
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        mask_mono = ((mask[:,:,0] > 254) + 0.0) + 2 * ((mask[:,:,1] > 254) + 0.0)  # we use rerd and green to encode different instances
        # instances are encoded as different colors
        obj_ids = np.unique(mask_mono)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask_mono == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class UniLabDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        jf = open(os.path.join(root, 'index.json'))
        index_json = json.load(jf)
        image_folders = index_json['frame_folders']

        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(root)))

        self.imgs = []
        files = []
        for folder in image_folders:
            cf = open(os.path.join(root, folder, 'files.csv'))
            #cf2 = open(os.path.join(root, folder, 'frame2video.csv'))
            index_csv = csv.DictReader(cf)
            #frame_csv = csv.DictReader(cf2)
            for row in index_csv:
                self.imgs.append(row)

            # for row in frame_csv:
            #     self.imgs.append(files[int(row['origin_frame'])])

            cf.close()

        jf.close()



    def __getitem__(self, idx):
        # load images and masks
        image_dict = self.imgs[idx]

        img_path = os.path.join(self.root, image_dict['file_loc'])
        #img = Image.open(img_path).convert("RGB")

        label = image_dict

        # if self.transforms is not None:
        #     img = self.transforms(img, label)

        return img_path, label

    def __len__(self):
        return len(self.imgs)




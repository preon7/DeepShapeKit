import math
import os

import cv2
import numpy as np
from tqdm import trange

from animal_model.MaskDatasets import Multiview_Dataset

def fish_video():
    multiview_data = Multiview_Dataset(root='../data/input/video_frames_30-07-2021')

    image_size = 512
    out = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'original_941-1195.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (image_size * 2, image_size))
    out1 = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'front_in_941-1195.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (image_size, image_size))
    out2 = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'bottom_in_941-1195.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (image_size, image_size))
    out3 = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'front_origin_941-1195.mp4'),
                           cv2.VideoWriter_fourcc(*'mp4v'), 20,
                           (2048, 1040))
    out4 = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'bottom_origin_941-1195.mp4'),
                           cv2.VideoWriter_fourcc(*'mp4v'), 20,
                           (2048, 1040))

    pbar = trange(200, desc="creating video")
    for sample_index in range(941,1195):
        sample = multiview_data[sample_index]

        if not sample['full_kpts']:
            continue

        im_path = sample["imgpaths"]
        bounding_boxes = sample["bboxes2"]
        # len_f = bounding_boxes[0,2] - bounding_boxes[0,0]
        padding_f_y = int((400 - bounding_boxes[0, 3] + bounding_boxes[0, 1]) / 2)
        padding_f_x = int((400 - bounding_boxes[0, 2] + bounding_boxes[0, 0]) / 2)
        # len_b = bounding_boxes[1, 2] - bounding_boxes[1, 0]
        padding_b_y = int((400 - bounding_boxes[1, 3] + bounding_boxes[1, 1]) / 2)
        padding_b_x = int((400 - bounding_boxes[1, 2] + bounding_boxes[1, 0]) / 2)

        front_img = cv2.imread(im_path[0])[bounding_boxes[0, 1] - padding_f_y:bounding_boxes[0, 3] + padding_f_y,
                    bounding_boxes[0, 0] - padding_f_x: bounding_boxes[0, 2] + padding_f_x]
        bottom_img = cv2.imread(im_path[1])[bounding_boxes[1, 1] - padding_b_y:bounding_boxes[1, 3] + padding_b_y,
                     bounding_boxes[1, 0] - padding_b_x: bounding_boxes[1, 2] + padding_b_x]
        bottom_img = cv2.rotate(bottom_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame = np.concatenate([cv2.resize(front_img, (image_size, image_size)),
                                cv2.resize(bottom_img, (image_size, image_size))], axis=1)

        out.write(frame)
        out1.write(cv2.resize(front_img, (image_size, image_size)))
        out2.write(cv2.resize(bottom_img, (image_size, image_size)))

        image = cv2.imread(im_path[0])
        bbox = bounding_boxes[0]
        image[bbox[1], bbox[0]:bbox[2]] = np.array([0, 255, 255])
        image[bbox[3], bbox[0]:bbox[2]] = np.array([0, 255, 255])
        image[bbox[1]:bbox[3], bbox[0]] = np.array([0, 255, 255])
        image[bbox[1]:bbox[3], bbox[2]] = np.array([0, 255, 255])

        bbox_2 = sample["bboxes"][0]
        image[bbox_2[1], bbox_2[0]:bbox_2[2]] = np.array([0, 255, 255])
        image[bbox_2[3], bbox_2[0]:bbox_2[2]] = np.array([0, 255, 255])
        image[bbox_2[1]:bbox_2[3], bbox_2[0]] = np.array([0, 255, 255])
        image[bbox_2[1]:bbox_2[3], bbox_2[2]] = np.array([0, 255, 255])

        out3.write(image)

        image_b = cv2.imread(im_path[1])
        bbox = bounding_boxes[1]
        image_b[bbox[1], bbox[0]:bbox[2]] = np.array([0, 255, 255])
        image_b[bbox[3], bbox[0]:bbox[2]] = np.array([0, 255, 255])
        image_b[bbox[1]:bbox[3], bbox[0]] = np.array([0, 255, 255])
        image_b[bbox[1]:bbox[3], bbox[2]] = np.array([0, 255, 255])

        bbox_2 = sample["bboxes"][1]
        image_b[bbox_2[1], bbox_2[0]:bbox_2[2]] = np.array([0, 255, 255])
        image_b[bbox_2[3], bbox_2[0]:bbox_2[2]] = np.array([0, 255, 255])
        image_b[bbox_2[1]:bbox_2[3], bbox_2[0]] = np.array([0, 255, 255])
        image_b[bbox_2[1]:bbox_2[3], bbox_2[2]] = np.array([0, 255, 255])

        out4.write(image_b)

        pbar.update(1)

    out.release()

def cropped_fish():
    multiview_data = Multiview_Dataset(root='../data/input/video_frames_30-07-2021')

    image_size = 512
    out1 = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'front_cropped_941-1195.mp4'),
                           cv2.VideoWriter_fourcc(*'mp4v'), 20,
                           (image_size, image_size))
    out2 = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'bottom_cropped_941-1195.mp4'),
                           cv2.VideoWriter_fourcc(*'mp4v'), 20,
                           (image_size, image_size))
    pbar = trange(200, desc="creating video")
    for sample_index in range(941, 1195):
        sample = multiview_data[sample_index]

        if not sample['full_kpts']:
            continue

        im_path = sample["imgpaths"]
        bbox_f = sample["bboxes"][0]
        bbox_b = sample["bboxes"][1]
        keypoints_f = sample["keypoints"][0]
        keypoints_b = sample["keypoints"][1]
        image_f = cv2.imread(im_path[0])
        image_b = cv2.imread(im_path[1])

        for j in range(keypoints_f.size(0)):
            #image_f[int(keypoints_f[j,1]), int(keypoints_f[j,0])] = np.array([0, 0, 255])
            cv2.circle(image_f, (int(keypoints_f[j,0]), int(keypoints_f[j,1])), 4, (0, 0, 255))

        for j in range(keypoints_b.size(0)):
            #image_b[int(keypoints_b[j,1]), int(keypoints_b[j,0])] = np.array([0, 0, 255])
            cv2.circle(image_b, (int(keypoints_b[j, 0]), int(keypoints_b[j, 1])), 4, (0, 0, 255))

        cropped_f = image_f[bbox_f[1]:bbox_f[3], bbox_f[0]:bbox_f[2]]
        cropped_b = image_b[bbox_b[1]:bbox_b[3], bbox_b[0]:bbox_b[2]]
        # origin_size = np.max([bbox[3] - bbox[1], bbox[2] - bbox[0]])

        (a, b, d) = cropped_f.shape
        if a > b:
            padding = ((0, 0), (math.floor((a - b) / 2.), math.ceil((a - b) / 2.)), (0, 0))
        else:
            padding = ((math.floor((b - a) / 2.), math.ceil((b - a) / 2.)), (0, 0), (0, 0))
        padded_f = np.pad(cropped_f, padding, mode='constant', constant_values=0)

        (a, b, d) = cropped_b.shape
        if a > b:
            padding = ((0, 0), (math.floor((a - b) / 2.), math.ceil((a - b) / 2.)), (0, 0))
        else:
            padding = ((math.floor((b - a) / 2.), math.ceil((b - a) / 2.)), (0, 0), (0, 0))
        padded_b = np.pad(cropped_b, padding, mode='constant', constant_values=0)

        out1.write(cv2.resize(padded_f, (image_size, image_size)))
        out2.write(cv2.resize(padded_b, (image_size, image_size)))

        pbar.update(1)

    out1.release()
    out2.release()

def fish_mask():
    multiview_data = Multiview_Dataset(root='../data/input/video_frames_30-07-2021')

    image_size = 512
    out1 = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'front_mask_941-1195.mp4'),
                           cv2.VideoWriter_fourcc(*'mp4v'), 20,
                           (image_size, image_size))
    out2 = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'bottom_mask_941-1195.mp4'),
                           cv2.VideoWriter_fourcc(*'mp4v'), 20,
                           (image_size, image_size))
    pbar = trange(200, desc="creating video")
    for sample_index in range(941, 1195):
        sample = multiview_data[sample_index]

        if not sample['full_kpts']:
            continue

        im_path = sample["imgpaths"]
        mask_f = cv2.cvtColor(sample['masks2'][0].numpy(),cv2.COLOR_GRAY2RGB)
        mask_b = cv2.cvtColor(sample['masks2'][1].numpy(),cv2.COLOR_GRAY2RGB)

        out1.write(cv2.resize(mask_f, (image_size, image_size)))
        out2.write(cv2.resize(mask_b, (image_size, image_size)))

        pbar.update(1)

    out1.release()
    out2.release()

def eye_video():
    image_size = 512
    out = cv2.VideoWriter(os.path.join('../data/output/multiview_demo/seq2video', 'eye_seq_large_941-1195_smooth.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'), 20,
                          (512, 512))#(2048, 1040))

    pbar = trange(200, desc="creating video")
    prev_img = None
    for i in range(941,1195):
        #img = cv2.imread("../data/output/multiview_demo/standard_images/eye_frames/frame_{}.png".format(i))
        img = cv2.imread("../data/output/multiview_demo/standard_images/{}_reverted.png".format(i))
        if img is not None:
            prev_img = img
        else:
            img = prev_img
        #out.write(cv2.resize(img, (image_size, image_size)))
        out.write(img)
        pbar.update(1)

    out.release()


if __name__ == '__main__':
    fish_video()
    #eye_video()
    #cropped_fish()
    #fish_mask()

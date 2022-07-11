import argparse
import csv
import json

import numpy as np

import torchtools.transforms as T
import torch.nn.functional as F
import animal_model.MaskDatasets as DFDataset
import model.MaskRCNN as MRCNN
import torch
import torchtools.engine as engine
import torchtools.utils as utils
import datetime
import os
from PIL import Image
from tqdm import trange

model_save_path = './model/saved_models'

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def eye_training():
    device = torch.device(0)  # if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2

    # use our dataset and defined transformations
    dataset = DFDataset.UniKNDataset('data/input/eye_detection_large', get_transform(train=True))
    dataset_test = DFDataset.UniKNDataset('data/input/eye_detection_large', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    dataset = torch.utils.data.Subset(dataset, indices[:-50])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = MRCNN.get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 20 epochs
    num_epochs = 30

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        engine.evaluate(model, data_loader_test, device=device)

        model.eval()
        with torch.no_grad():
            test_image_batch, test_image_labels = next(iter(data_loader_test))
            test_image_batch = list(img.to(device) for img in test_image_batch)
            test_mask_batch = model(test_image_batch)

        for i in range(len(test_image_batch)):
            original = Image.fromarray(test_image_batch[i].mul(255).permute(1, 2, 0).byte().cpu().numpy())
            predicted_mask = Image.fromarray(test_mask_batch[i]['masks'][0, 0].mul(255).byte().cpu().numpy())
            original.save('./data/output/eye_detect_test/image_{}_ep{}.png'.format(i, epoch))
            predicted_mask.save('./data/output/eye_detect_test/mask_{}_ep{}.png'.format(i, epoch))

    model_name = 'eye_detection_model_' + datetime.date.today().isoformat()
    torch.save(model.state_dict(), model_save_path + '/' + model_name)

def main(argv):

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(0) #if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2

    if argv[0] == '-t':

        # use our dataset and defined transformations
        # dataset = DFDataset.UniKNDataset('data/input/combined_mask_training_set', get_transform(train=True))
        dataset = DFDataset.DeepFishDataset('data/input/DeepFish/Segmentation', get_transform(train=True))
        dataset_test = DFDataset.UniKNDataset('data/input/combined_mask_training_set', get_transform(train=False))
        # coco_dataset = DFDataset.CocoDetection('data/input/coco_custom/train', 'data/input/coco_custom/train/train_coco.json', get_transform(train=True))

        # split the dataset in train and test set
        # indices = torch.randperm(len(dataset)).tolist()
        # dataset = torch.utils.data.Subset(dataset, indices[:-50])
        # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        indices = torch.randperm(len(dataset_test)).tolist()
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=0,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)

        # get the model using our helper function
        model = MRCNN.get_model_instance_segmentation(num_classes)

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's train it for 20 epochs
        num_epochs = 100

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            engine.evaluate(model, data_loader_test, device=device)

            model.eval()
            with torch.no_grad():
                test_image_batch, test_image_labels = next(iter(data_loader_test))
                test_image_batch = list(img.to(device) for img in test_image_batch)
                test_mask_batch = model(test_image_batch)

            for i in range(len(test_image_batch)):

                original = Image.fromarray(test_image_batch[i].mul(255).permute(1, 2, 0).byte().cpu().numpy())
                predicted_mask = Image.fromarray(test_mask_batch[i]['masks'][0, 0].mul(255).byte().cpu().numpy())
                original.save('./data/output/mask_rcnn_test_result_kn/image_{}_ep{}.png'.format(i, epoch))
                predicted_mask.save('./data/output/mask_rcnn_test_result_kn/mask_{}_ep{}.png'.format(i, epoch))


        model_name = 'kn_segmentation_model_' + datetime.date.today().isoformat()
        torch.save(model.state_dict(), model_save_path + '/' + model_name)
    elif argv[0] == '-m':
        model_path = argv[1]

        # get the model using our helper function
        model = MRCNN.get_model_instance_segmentation(num_classes)

        # move model to the right device
        model.to(device)
        model.load_state_dict(torch.load(model_path))

        dataset = DFDataset.UniLabDataset('data/input/reconstruction_set/bottom_no_overlap/', get_transform(train=False))
        data_loader_test = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)

        model.eval()
        with torch.no_grad():
            k = 0
            for images, targets in data_loader_test:
                images = images[0]
                #print(images[0][0])
                crop_image = images[0][0].mul(255).permute(1,0)

                image_width = crop_image.size()[1]
                image_height = crop_image.size()[0]

                #original = Image.fromarray(images[0][0].mul(255).cpu().byte().numpy())
                #original.save('./data/output/lab_video_crop_front/image_o_{}.png'.format(k))

                images = list(torch.unsqueeze(img.to(device), 2).permute(2,0,1) for img in images[0])
                predictions = model(images)

                for i in range(predictions[0]['boxes'].size()[0]):
                    # only 2 fishes in the scene
                    if i > 1:
                        break

                    bounding_box = predictions[0]['boxes'][i].data.int().cpu().numpy()  # [w_low, h_low, w_up, h_up]
                    cropped = crop_image[int(bounding_box[0]):int(bounding_box[2]),
                                         int(bounding_box[1]):int(bounding_box[3])]

                    crop_mask = predictions[0]['masks'][i, 0].mul(255).permute(1, 0)
                    cropped_mask = crop_mask[int(bounding_box[0]):int(bounding_box[2]),
                              int(bounding_box[1]):int(bounding_box[3])]


                    diff = abs(bounding_box[2] - bounding_box[0] - (bounding_box[3] - bounding_box[1]))

                    output = cropped
                    out_mask = cropped_mask

                    if bounding_box[2] - bounding_box[0] < bounding_box[3] - bounding_box[1]:
                        # padding height
                        output = F.pad(input=cropped,
                                       pad=(0,0,int(diff / 2.0),
                                            int(diff / 2.0)),
                                       mode='constant', value=0)
                        out_mask = F.pad(input=cropped_mask,
                                       pad=(0,0,int(diff / 2.0),
                                            int(diff / 2.0)),
                                       mode='constant', value=0)

                    if bounding_box[2] - bounding_box[0] > bounding_box[3] - bounding_box[1]:
                        # padding height
                        output = F.pad(input=cropped,
                                       pad=(int(diff / 2.0),
                                            int(diff / 2.0),0,0),
                                       mode='constant', value=0)
                        out_mask = F.pad(input=cropped_mask,
                                       pad=(int(diff / 2.0),
                                            int(diff / 2.0), 0, 0),
                                       mode='constant', value=0)

                    output = Image.fromarray(output.permute(1,0).cpu().byte().numpy())
                    output.save('./data/output/lab_nooverlap_crop_bottom/image_{}_{}.png'.format(k, i))
                    predicted_mask = Image.fromarray(out_mask.permute(1,0).byte().cpu().numpy())
                    predicted_mask.save('./data/output/lab_nooverlap_crop_bottom/image_{}_{}_mask.png'.format(k, i))

                #for i in range(len(images)):
                    # save masks
                    # original = Image.fromarray(images[i].mul(255).permute(1, 2, 0).byte().cpu().numpy())
                    # predicted_mask = Image.fromarray(test_mask_batch[i]['masks'][0, 0].mul(255).byte().cpu().numpy())
                    # original.save('./data/output/lab_video_test_result_top/image_{}.png'.format(k))
                    # predicted_mask.save('./data/output/lab_video_test_result_top/mask_{}.png'.format(k))
                print('processing image {}'.format(k), end='\r')
                k += 1
        print('\n finish')


def train_model(dataset_path, model_save_path, out_dir, num_epochs, bs, lr, device):
    # train on the GPU or on the CPU, if a GPU is not available
    # device = torch.device(1)  # if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2

    # use our dataset and defined transformations
    dataset = DFDataset.UniKNDataset(dataset_path, get_transform(train=True))
    dataset_test = DFDataset.UniKNDataset(dataset_path, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = MRCNN.get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 20 epochs
    # num_epochs = 20
    test_iter = iter(data_loader_test)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        engine.evaluate(model, data_loader_test, device=device)

        model.eval()
        with torch.no_grad():
            test_image_batch, test_image_labels = next(test_iter)
            test_image_batch = list(img.to(device) for img in test_image_batch)
            test_mask_batch = model(test_image_batch)

        for i in range(len(test_image_batch)):
            if test_mask_batch[i]['masks'].size(0) == 0:
                print('no detection in test')
                continue
            original = Image.fromarray(test_image_batch[i].mul(255).permute(1, 2, 0).byte().cpu().numpy())
            predicted_mask = Image.fromarray(test_mask_batch[i]['masks'][0, 0].mul(255).byte().cpu().numpy())
            original.save(os.path.join(out_dir, 'image_{}_ep{}.png'.format(i, epoch)))
            predicted_mask.save(os.path.join(out_dir, 'mask_{}_ep{}.png'.format(i, epoch)))

    torch.save(model.state_dict(), model_save_path)


def predict(dataset_path, model_path, device, num_classes=2):
    # get the model using our helper function
    model = MRCNN.get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    dataset = DFDataset.UniLabDataset(dataset_path, get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # create csv index files in each video folder
    jf = open(os.path.join(dataset_path, 'index.json'))
    index_json = json.load(jf)
    image_folders = index_json['frame_folders']

    csvwriters = {}
    for folder in image_folders:
        if not os.path.exists(os.path.join(dataset_path, folder, 'cropped')):
            os.mkdir(os.path.join(dataset_path, folder, 'cropped'))

        if not os.path.exists(os.path.join(dataset_path, folder, 'mask')):
            os.mkdir(os.path.join(dataset_path, folder, 'mask'))

        csv_out_file = open(os.path.join(dataset_path, folder, 'files_crop.csv'), 'w')
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                               quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder', 'bbox'])

        csvwriters[folder] = csvwriter

    # frames contain no overlap
    # target_frames = list(range(3718, 4384)) + list(range(5025, 5555)) + list(range(5825, 5908)) + \
    #                 list(range(6124, 6309)) + list(range(6577, 6696)) + list(range(11751, 11813)) +\
    #                 list(range(13946, 14237)) + list(range(16046, 16257))
    target_frames = list(range(index_json['image_count']))

    model.eval()
    with torch.no_grad():
        k = 0
        pbar = trange(len(target_frames), desc="detect from frames")
        for image, label in data_loader_test:
            # only use non-overlap frames
            label = label[0]
            if int(label['frame']) not in target_frames:
                continue

            pbar.set_description('detecting frame {}'.format(label['frame']))
            images = torch.from_numpy(np.array(Image.open(image[0]).convert("RGB"))) #image[0]
            csvwriter = csvwriters[label['folder']]

            # print(images[0][0])
            crop_image = images.permute(1,0,2) #[0][0].mul(255).permute(1, 0)

            # image_width = crop_image.size()[1]
            # image_height = crop_image.size()[0]

            # original = Image.fromarray(images[0][0].mul(255).cpu().byte().numpy())
            # original.save('./data/output/lab_video_crop_front/image_o_{}.png'.format(k))

            #images = list(torch.unsqueeze(img.to(device), 2).permute(2, 0, 1) for img in images[0])
            images = [images.to(device).permute(2, 0, 1) / 255.]
            predictions = model(images)

            for i in range(predictions[0]['boxes'].size()[0]):
                # only 2 fishes in the scene
                if i > 1:
                    break
                mask = predictions[0]['masks'][i, 0].cpu().numpy()

                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                bounding_box = [xmin, ymin, xmax, ymax]

                #bounding_box = predictions[0]['boxes'][i].data.int().cpu().numpy()  # [w_low, h_low, w_up, h_up]
                cropped = crop_image[int(bounding_box[0]):int(bounding_box[2]),
                          int(bounding_box[1]):int(bounding_box[3])]
                # cropped = crop_image[int(bounding_box[1]):int(bounding_box[3]),
                #                     int(bounding_box[0]):int(bounding_box[2])]

                crop_mask = predictions[0]['masks'][i, 0].mul(255).permute(1, 0)
                cropped_mask = crop_mask[int(bounding_box[0]):int(bounding_box[2]),
                               int(bounding_box[1]):int(bounding_box[3])]

                diff = abs(bounding_box[2] - bounding_box[0] - (bounding_box[3] - bounding_box[1]))

                output = cropped.permute(2,0,1)
                out_mask = cropped_mask

                if bounding_box[2] - bounding_box[0] < bounding_box[3] - bounding_box[1]:
                    # padding height
                    output = F.pad(input=output,
                                   pad=(0, 0, int(diff / 2.0),
                                        int(diff / 2.0)),
                                   mode='constant', value=0)
                    out_mask = F.pad(input=cropped_mask,
                                     pad=(0, 0, int(diff / 2.0),
                                          int(diff / 2.0)),
                                     mode='constant', value=0)

                if bounding_box[2] - bounding_box[0] > bounding_box[3] - bounding_box[1]:
                    # padding height
                    output = F.pad(input=output,
                                   pad=(int(diff / 2.0),
                                        int(diff / 2.0), 0, 0),
                                   mode='constant', value=0)
                    out_mask = F.pad(input=cropped_mask,
                                     pad=(int(diff / 2.0),
                                          int(diff / 2.0), 0, 0),
                                     mode='constant', value=0)

                crop_out_dir = os.path.join(dataset_path, label['folder'], 'cropped')
                mask_out_dir = os.path.join(dataset_path, label['folder'], 'mask')
                mask_full_dir = os.path.join(dataset_path, label['folder'], 'mask_full')

                output = Image.fromarray(output.permute(2, 1, 0).cpu().byte().numpy())
                output.save(os.path.join(crop_out_dir, 'image_{}_{}.png'.format(k, i)))
                predicted_mask = Image.fromarray(out_mask.permute(1, 0).byte().cpu().numpy())
                predicted_mask.save(os.path.join(mask_out_dir, 'image_{}_{}_mask.png'.format(k, i)))
                full_mask = Image.fromarray(predictions[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
                full_mask.save(os.path.join(mask_full_dir, 'image_{}_{}_mask.png'.format(k, i)))

                crop_out_dir = os.path.join(crop_out_dir, 'image_{}_{}.png'.format(k, i))
                mask_out_dir = os.path.join(mask_out_dir, 'image_{}_{}_mask.png'.format(k, i))

                csvwriter.writerow([label['frame'],
                                    '/'.join(crop_out_dir.split('/')[-3:]),
                                    'cropped',
                                    str(i),
                                    label['folder'],
                                    str(bounding_box)])

                csvwriter.writerow([label['frame'],
                                    '/'.join(mask_out_dir.split('/')[-3:]),
                                    'mask',
                                    str(i),
                                    label['folder'],
                                    str(bounding_box)])

            # for i in range(len(images)):
            # save masks
            # original = Image.fromarray(images[i].mul(255).permute(1, 2, 0).byte().cpu().numpy())
            # predicted_mask = Image.fromarray(test_mask_batch[i]['masks'][0, 0].mul(255).byte().cpu().numpy())
            # original.save('./data/output/lab_video_test_result_top/image_{}.png'.format(k))
            # predicted_mask.save('./data/output/lab_video_test_result_top/mask_{}.png'.format(k))
            pbar.update(1)
            k += 1
    print('\n finish')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a maskRCNN or process a set of images with a trained model')
    parser.add_argument('-t', '--train', type=bool, default=False)
    parser.add_argument('-p', '--model_path', type=str, default="model/saved_models/")
    parser.add_argument('-m', '--model_name', type=str, default="kn_segmentation_model_{}".format(datetime.datetime.now().strftime("%Y-%m-%d")))
    parser.add_argument('-s', '--save_dir', type=str, default="data/output/test_save")
    # parser.add_argument('-o', '--out_dir', type=str, default="data/output/prediction_output")

    parser.add_argument('--device', type=str, default='default')
    parser.add_argument('-d', '--dataset_path', type=str, default='data/input/combined_mask_training_set')
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('-e', '--num_epochs', type=int, default=20)

    eye_training()

    #main(['-t'])
    # predict('/home/liang/Documents/ruiheng/master2021/data/input/video_frames_30-07-2021/',
    #         'model/saved_models/kn_segmentation_model_2022-01-13', 'cuda')

    # args = parser.parse_args()
    #
    # model_full_path = os.path.join(args.model_path, args.model_name)
    #
    # if args.device == 'default':
    #     device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    # else:
    #     device = torch.device(args.device)
    #
    # if args.train:
    #     if not os.path.exists(args.dataset_path):
    #         raise Exception('dataset path does not exist')
    #     if not os.path.exists(args.model_path):
    #         raise Exception('model output path does not exist')
    #     if not os.path.exists(args.save_dir):
    #         raise Exception('test result save path does not exist')
    #
    #     train_model(args.dataset_path, model_full_path, args.save_dir, args.num_epochs,
    #                 args.batch_size, args.learning_rate, device)
    # else:
    #     if not os.path.exists(model_full_path):
    #         raise Exception('input model does not exist')
    #     if not os.path.exists(args.dataset_path):
    #         raise Exception('dataset path does not exist')
    #     # if not os.path.exists(args.out_dir):
    #     #     raise Exception('image output path does not exist')
    #
    #     predict(args.dataset_path, model_full_path, device)

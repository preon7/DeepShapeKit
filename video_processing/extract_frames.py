import cv2
import sys
import csv
import json
import os
import argparse
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract frames from videos and create index file')
    parser.add_argument('-v', '--videos', nargs='+', type=str, default="")
    parser.add_argument('-o', '--outdir', type=str, default="../data/input")
    args = parser.parse_args()

    video_list = args.videos
    dist = args.outdir

    if not os.path.exists(dist):
        raise Exception('Output dir do not exist')

    dt = datetime.datetime.now()
    dt = dt.strftime("%d-%m-%Y")
    dist = os.path.join(dist, 'video_frames_' + dt)
    if not os.path.exists(dist):
        os.mkdir(dist)

    json_out_file = open(os.path.join(dist, 'index.json'), "w")
    json_index = {}
    json_index['frame_folders'] = []
    csv_list = {}

    for video_path in video_list:
        if not os.path.exists(video_path):
            raise Exception('Input video do not exist')

        print("processing video: {}".format(video_path))

        video_name = video_path.split('/')[-1]
        ext_len = len(video_name.split('.')[-1])
        save_path = os.path.join(dist, video_name[:-ext_len-1], 'origin')
        if not os.path.exists(os.path.join(dist, video_name[:-ext_len-1])):
            os.mkdir(os.path.join(dist, video_name[:-ext_len - 1]))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        csv_out_file = open(os.path.join(dist, video_name[:-ext_len-1], 'files.csv'), 'w')
        csvwriter = csv.writer(csv_out_file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['frame', 'file_loc', 'category', 'sub_index', 'folder'])

        # cv2 extract frames
        capture = cv2.VideoCapture(video_path)

        image_count = 0
        frame_number = 0
        success = True

        while capture.isOpened() and success:
            success, frame = capture.read()

            if success:
                file_loc = os.path.join(save_path, "{}_{}.png".format(video_name[:-ext_len-1], frame_number))
                cv2.imwrite(file_loc, frame)
                image_count += 1
                csvwriter.writerow([frame_number,
                                    os.path.join(video_name[:-ext_len-1], "origin/{}_{}.png".format(video_name[:-ext_len-1], frame_number)),
                                    'origin',
                                    0,
                                    video_name[:-ext_len-1]])

            frame_number += 1
            print('frame out: {}, total image: {}'.format(frame_number, image_count), end='\r')

            # if image_count > 5:
            #     break

        print('total image: {}, done                  '.format(image_count))

        json_index['frame_folders'].append(video_name[:-ext_len-1])
        csv_list[video_name[:-ext_len-1]] = os.path.join(dist, video_name[:-ext_len-1], 'files.csv')

        csv_out_file.close()
        capture.release()

    json_index['status'] = 'origin'
    json_index['index_files'] = csv_list
    json_index['image_count'] = image_count

    json.dump(json_index, json_out_file)
    json_out_file.close()

# capture = cv2.VideoCapture('../data/detection_1/clipped_front.mp4')
#
# image_count = 0
# image_id = 0
# success = True
# # target_frames = list(range(3718, 4384)) + list(range(5025, 5555)) + list(range(5825, 5908)) + \
# #                 list(range(6124, 6309)) + list(range(6577, 6696)) + list(range(11751, 11813)) +\
# #                 list(range(13946, 14237)) + list(range(16046, 16257))
# while capture.isOpened() and success:
#   success, frame = capture.read()
#
#   # save every 10 frame
#   #if success and image_count % 10 == 0:
#   # if success and image_count in target_frames:
#   if success:
#     cv2.imwrite('../data/detection_1/frames/front/front_frame_{}.jpg'.format(image_id), frame)
#     image_id += 1
#
#   image_count += 1
#   # print('image out: {}, total image: {}, {}'.format(image_id, image_count, image_count in target_frames), end='\r')
#   print('image out: {}, total image: {}'.format(image_id, image_count), end='\r')
#
# print('total image: {}, finished'.format(image_id))
#
# capture.release()
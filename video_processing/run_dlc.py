import deeplabcut

def create_proj(config_path=None):
    if not config_path:
        config_path = deeplabcut.create_new_project('master2021demo_front', 'Ruiheng Wu',
                    ['/Users/ruihengwu/Documents/Study/Master thesis/master_project/data/input/sample-front.mp4'],
                    copy_videos=False, multianimal=True)

    print(config_path)

    return config_path

def add_videos(list_of_video_paths):
    pass
    #deeplabcut.add_new_videos('Full path of the project configuration file*',
                              #['full path of video 4', 'full path of video 5'], copy_videos=True / False)

def extract_frames(config_path):
    deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False,
                              crop=False)

# if __name__ == '__main__':
#     #cf_path = '/Users/ruihengwu/Documents/Study/Master thesis/master_project/video_processing/master2021demo-Ruiheng Wu-2021-05-25/config.yaml'
#
#     # cf_path = '/Users/ruihengwu/Documents/Study/Master thesis/master_project/video_processing/master2021demo_bottom-Ruiheng Wu-2021-06-01/config.yaml'
#
#     cf_path = '/Users/ruihengwu/Documents/Study/Master thesis/master_project/video_processing/master2021demo_front-Ruiheng Wu-2021-06-02/config.yaml'
#     # cf_path = create_proj()
#
#     # extract_frames(cf_path)
#     # deeplabcut.label_frames(cf_path)
#     # deeplabcut.check_labels(cf_path, visualizeindividuals=True)
#     deeplabcut.cropimagesandlabels(cf_path, size=(600, 600), userfeedback=False)


if __name__ == '__main__':
    # config_path = deeplabcut.create_new_project('hiwi2021_goldfish_front', 'Ruiheng_Wu',
    #                                             ['/Users/ruihengwu/Documents/work/2021-09/video/goldfish/front/000000.mp4'],
    #                                             copy_videos=False, multianimal=True)
    # print(config_path)
    # config_path = deeplabcut.create_new_project('hiwi2021_goldfish_bottom', 'Ruiheng_Wu',
    #                                             ['/Users/ruihengwu/Documents/work/2021-09/video/goldfish/bottom/000000-b.mp4'],
    #                                             copy_videos=False, multianimal=True)
    # print(config_path)

    config_path = '/Users/ruihengwu/Documents/Study/Master thesis/master_project/video_processing/hiwi2021_goldfish_front-Ruiheng_Wu-2021-11-03/config.yaml'
    # config_path = '/Users/ruihengwu/Documents/Study/Master thesis/master_project/video_processing/hiwi2021_goldfish_bottom-Ruiheng_Wu-2021-11-08/config.yaml'

    # extract_frames(config_path)
    # deeplabcut.label_frames(config_path)
    deeplabcut.cropimagesandlabels(config_path, size=(700, 700), userfeedback=False)

# if __name__ == '__main__':
#     # config_path = deeplabcut.create_new_project('hiwi2021_sunbleak_front', 'Ruiheng_Wu',
#     #                                             ['/Users/ruihengwu/Documents/work/2021-09/video/sunbleak/front/000000-f.mp4',
#     #                                              '/Users/ruihengwu/Documents/work/2021-09/video/sunbleak/front/000000-f2.mp4'],
#     #                                             copy_videos=False, multianimal=True)
#     # print(config_path)
#     # config_path = deeplabcut.create_new_project('hiwi2021_sunbleak_bottom', 'Ruiheng_Wu',
#     #                                             ['/Users/ruihengwu/Documents/work/2021-09/video/sunbleak/bottom/000000-b.mp4',
#     #                                              '/Users/ruihengwu/Documents/work/2021-09/video/sunbleak/bottom/000000-b2.mp4'],
#     #                                             copy_videos=False, multianimal=True)
#     # print(config_path)
#
#     # config_path = '/Users/ruihengwu/Documents/Study/Master thesis/master_project/video_processing/hiwi2021_sunbleak_front-Ruiheng_Wu-2021-11-11/config.yaml'
#     config_path = '/Users/ruihengwu/Documents/Study/Master thesis/master_project/video_processing/hiwi2021_sunbleak_bottom-Ruiheng_Wu-2021-11-15/config.yaml'
#
#     # extract_frames(config_path)
#     # deeplabcut.label_frames(config_path)
#     deeplabcut.cropimagesandlabels(config_path, size=(700, 700), userfeedback=False)
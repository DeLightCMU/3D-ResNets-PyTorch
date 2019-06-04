import os
import json
import random

dataset_path = '/data/Kinetics400'
annotation_path = '/data/Kinetics400/annotation_400/'
splits = ['val', 'train']

trainvallist_path = os.path.join(dataset_path, 'trainvalList_test')
if not os.path.exists(trainvallist_path):
    os.makedirs(trainvallist_path)

with open('/data/Kinetics400/trainvalList_test/classInd.txt', 'r') as f:
    lines = f.readlines()
class_name_list = []
for class_name in lines:
    class_name_list.append(class_name.split('\r')[0])
class_name_list = [item.replace("\n", "").split(' ')[1] for item in class_name_list]


for idx_split, split in enumerate(splits):
    with open(os.path.join(annotation_path, 'kinetics_' + split, 'kinetics_' + split + '.json'), 'rb') as f:
        gt = json.load(f, encoding='utf-8')

    f = open(os.path.join(trainvallist_path, split + '_imglist_all_.cvs'), 'w')
    f.write('{0}, {1}\n'.format('image_name', 'class'))
    # fzero = open(os.path.join(trainvallist_path, split + 'img_zerolist.txt'), 'w')
    # flost = open(os.path.join(trainvallist_path, split + 'img_lostlist.txt'), 'w')
    count = 0
    lost_count = 0
    zero_count = 0

    video_names = list(gt.keys())
    random.shuffle(video_names)

    class_appearance = []
    for video_name in video_names:
        video_anno = gt[video_name]
        class_name = video_anno['annotations']['label'].replace(' ', '_')
        video_name = '{0}_{1:06d}_{2:06d}'.format(video_name, int(video_anno['annotations']['segment'][0]),
                                                  int(video_anno['annotations']['segment'][1]))

        if os.path.exists(os.path.join(dataset_path, split, class_name, video_name)):
            file_list = os.listdir(os.path.join(dataset_path, split, class_name, video_name))
            if len(file_list) == 1:
                # fzero.write('{0} {1}\n'.format(video_name, class_name_list.index(class_name)))
                zero_count = zero_count + 1
            else:
                if not class_name in class_appearance:
                    class_appearance.append(class_name)
                for idx, file_name in enumerate(file_list):
                    if '.jpg' in file_name and idx % 30 == 0:
                        f.write('{0}, {1}\n'.format(os.path.join(dataset_path, split, class_name, video_name, file_name),
                                                   class_name_list.index(class_name)))
                count = count + 1
        else:
            # flost.write('{0} {1}\n'.format(video_name, class_name_list.index(class_name)))
            lost_count = lost_count + 1

        # if count > 2000 and len(class_appearance) == 400:
        #     print(split + 'count:' + str(count))
        #     print(split + 'lost:' + str(lost_count))
        #     print(split + 'zero:' + str(zero_count))
        #     break

    print(split + 'count:' + str(count))
    print(split + 'lost:' + str(lost_count))
    print(split + 'zero:' + str(zero_count))




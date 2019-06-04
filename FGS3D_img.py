import torch.nn as nn
import torch

from models.resnet import resnet101
from models.FlowNetS import flownets
from models.inception3D import InceptionModule, Unit3D
from models.warping import warp
from collections import OrderedDict


class FGS3DIMG(nn.Module):

    def __init__(self, num_classes=400, num_frames=64, num_keyframe=8, dropout_keep_prob=0.5):
        super(FGS3DIMG, self).__init__()

        self.num_frames = num_frames
        self.num_keyframe = num_keyframe
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob

        ##############################################
        # Load resnet model
        ##############################################
        self.resnet_feature = resnet101(pretrained=False)
        num_ftrs = self.resnet_feature.fc.in_features
        self.resnet_feature.fc = nn.Linear(num_ftrs, num_classes)
        ResNet_state_dict = torch.load('/data/Kinetics400/result/ResNetImg_lr0.00025/F90epochs/save_145.pth')
        ResNet_state_dict = ResNet_state_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in ResNet_state_dict.items():
            name = k[22:]  # remove `module.`
            new_state_dict[name] = v
        self.resnet_feature.load_state_dict(new_state_dict)

        print('debug')


    def forward(self, x):
        # x: [batchisze/n_gpu 3 64 112 112]

        # extract features for key frames
        self.resnet_feature.eval()
        feature_keyframe, pred_keyframes = self.resnet_feature(x)  # [8 1024 14 14] [8 400]

        return pred_keyframes, pred_keyframes

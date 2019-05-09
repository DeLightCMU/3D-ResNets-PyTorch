import torch.nn as nn
import torch

from models.resnet import resnet34
from models.FlowNetS import flownets
from models.Inception import inception_3D
from models.warping import warp

class FGS3D(nn.Module):

    def __init__(self, num_classes=400, input_channel=3, num_frames=64, num_keyframe=8):
        super(FGS3D, self).__init__()


        self.resnet_feature = resnet34(pretrained=True)
        self.optical_flow = flownets()
        self.inception_3D_1 = inception_3D()
        self.inception_3D_2 = inception_3D()
        self.inception_3D_3 = inception_3D()

        self.softmax_cls = nn.Softmax()
        self.softmax_img = nn.Softmax()

    def forward(self, x):
        # x: [1 64 3 224 224]

        # slice key frames
        x0 = torch.index_select(x, 1, 0)  # [1 1 3 224 224]
        x1 = torch.index_select(x, 1, 8)
        x2 = torch.index_select(x, 1, 16)
        x3 = torch.index_select(x, 1, 24)
        x4 = torch.index_select(x, 1, 32)
        x5 = torch.index_select(x, 1, 40)
        x6 = torch.index_select(x, 1, 48)
        x7 = torch.index_select(x, 1, 56)


        # concat key frames
        x_keyframes = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7), 1)  # [1 8 3 224 224]

        # reshape [8 3 224 224]
        x_keyframes = torch.squeeze(x_keyframes)  # [8 3 224 224]

        # extract features for key frames
        feature_keyframe, pred_keyframes = self.resnet_feature(x_keyframes)
        pred_keyframes = self.softmax_cls(pred_keyframes)


        # slice other frames
        # to do


        # compute optical flow
        # to do


        # warping
        # to do


        # concat feature
        # to do
        feature = torch.cat()

        # 3D inception
        feature = self.inception_3D_1(feature)
        feature = self.inception_3D_2(feature)
        feature = self.inception_3D_3(feature)


        # loss for video
        pred_video = self.softmax_img(feature)



        return pred_keyframes, pred_video


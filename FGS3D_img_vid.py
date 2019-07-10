import torch.nn as nn
import torch

from models.resnet import resnet101
from models.FlowNetS import flownets
from models.inception3D import InceptionModule, Unit3D
from models.warping import warp
from collections import OrderedDict

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

class FGS3D(nn.Module):

    def __init__(self, num_classes=400, num_frames=64, num_keyframe=8, dropout_keep_prob=0.5):
        super(FGS3D, self).__init__()

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
        """
        ResNet_state_dict = torch.load('/data/Kinetics400/result/ResNetImg_lr0.00025/F90epochs/save_145.pth')
        ResNet_state_dict = ResNet_state_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in ResNet_state_dict.items():
            name = k[22:]  # remove `module.`
            new_state_dict[name] = v
        self.resnet_feature.load_state_dict(new_state_dict)
        set_parameter_requires_grad(self.resnet_feature)
        """

        self.feat_conv_3x3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=6, dilation=6)
        torch.nn.init.normal_(self.feat_conv_3x3.weight, mean=0., std=0.01)
        torch.nn.init.constant_(self.feat_conv_3x3.bias, 0.0)
        self.feat_conv_3x3_relu = nn.ReLU(inplace=True)

        self.logits_img_vid = nn.Linear(8*400, self.num_classes)
        torch.nn.init.normal_(self.logits_img_vid.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.logits_img_vid.bias, 0.0)

        state_dict = torch.load('/data/Kinetics400/result/img_vid/save_20.pth')
        state_dict = state_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict)
        set_parameter_requires_grad(self.resnet_feature)


    def forward(self, x):

        # x: [batchisze/n_gpu 3 64 112 112]

        num_mini_clips = int(self.num_keyframe)
        lenght_mini_clip = int(self.num_frames / self.num_keyframe)

        ###############################################################
        # data preparing
        # slice key frames
        x_trunk = torch.split(x, 1, dim=2)  # x_trunk: 64 * [1 3 1 224 224]


        # data_bef = torch.cat(x_trunk[0:-2], dim=2) # data_bef: [1 3 62 224 224]
        # data_curr = torch.cat(x_trunk[1:-1], dim=2)  # data_curr: [1 3 62 224 224]
        # data_aft = torch.cat(x_trunk[2:], dim=2)  # data_aft: [1 3 62 224 224]


        # key frames
        data_key1 = x_trunk[0]                        # data_key1: [1 3 1 224 224]
        data_key2 = x_trunk[0 + lenght_mini_clip * 1] # data_key2: [1 3 1 224 224]
        data_key3 = x_trunk[0 + lenght_mini_clip * 2]
        data_key4 = x_trunk[0 + lenght_mini_clip * 3]
        data_key5 = x_trunk[0 + lenght_mini_clip * 4]
        data_key6 = x_trunk[0 + lenght_mini_clip * 5]
        data_key7 = x_trunk[0 + lenght_mini_clip * 6]
        data_key8 = x_trunk[0 + lenght_mini_clip * 7]

        # No key frames
        nokey1 = torch.cat(x_trunk[1:lenght_mini_clip * 1], dim=2)                         # nokey1: [1 3 7 224 224]
        nokey2 = torch.cat(x_trunk[1 + lenght_mini_clip * 1:lenght_mini_clip * 2], dim=2)  # nokey1: [1 3 7 224 224]
        nokey3 = torch.cat(x_trunk[1 + lenght_mini_clip * 2:lenght_mini_clip * 3], dim=2)
        nokey4 = torch.cat(x_trunk[1 + lenght_mini_clip * 3:lenght_mini_clip * 4], dim=2)
        nokey5 = torch.cat(x_trunk[1 + lenght_mini_clip * 4:lenght_mini_clip * 5], dim=2)
        nokey6 = torch.cat(x_trunk[1 + lenght_mini_clip * 5:lenght_mini_clip * 6], dim=2)
        nokey7 = torch.cat(x_trunk[1 + lenght_mini_clip * 6:lenght_mini_clip * 7], dim=2)
        nokey8 = torch.cat(x_trunk[1 + lenght_mini_clip * 7:], dim=2)


        ###############################################################
        # processing for key frames
        # concat key frames
        x_keyframes = torch.cat(x_trunk[0::8], dim=2)  # [1 3 8 224 224]

        # reshape [8 3 224 224]
        x_keyframes = torch.squeeze(x_keyframes)  # [3 8 224 224]
        x_keyframes = x_keyframes.permute(1, 0, 2, 3)

        # extract features for key frames
        self.resnet_feature.eval()
        feature_keyframe, pred_keyframes = self.resnet_feature(x_keyframes)   # [8 1024 14 14] [8 400]

        # flatten
        feat_img_flat = torch.flatten(pred_keyframes)

        # prediction
        pred_img_video = self.logits_img_vid(feat_img_flat)

        # # loss for video
        # pred_video = self.softmax_cls(pred_video)
        pred_video = torch.unsqueeze(pred_img_video, dim=0)

        return pred_video


    def warping_function(self, flow, feat_cam, duration):

        feat_keys = torch.cat((feat_cam, feat_cam, feat_cam, feat_cam, feat_cam, feat_cam, feat_cam), dim=0)

        return warp(feat_keys, flow)  # [7 512 7 7]


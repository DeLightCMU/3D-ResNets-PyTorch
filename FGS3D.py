import torch.nn as nn
import torch

from models.resnet import resnet34
from models.FlowNetS import flownets
from models.inception3D import InceptionModule, Unit3D
from models.warping import warp


class FGS3D(nn.Module):

    def __init__(self, num_classes=400, num_frames=64, num_keyframe=8, dropout_keep_prob=0.5):
        super(FGS3D, self).__init__()

        self.num_frames = num_frames
        self.num_keyframe = num_keyframe
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob

        self.resnet_feature = resnet34(pretrained=True, num_classes=400)

        self.flownetresize = nn.AvgPool2d(kernel_size=8, stride=8)
        FlowNet_state_dict = torch.load('/home/weik/pretrainedmodels/FlowNetS/flownets_from_caffe.pth.tar.pth')
        self.flownets = flownets(FlowNet_state_dict)

        self.inception_3D_1 = InceptionModule(512, [192,96,208,16,48,64], 'mixed_4a')
        self.inception_3D_2 = InceptionModule(192+208+48+64, [160,112,224,24,64,64], 'mixed_4b')
        self.inception_3D_3 = InceptionModule(160+224+64+64, [128,128,256,24,64,64], 'mixed_4c')

        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],

                                     stride=(2, 1, 1))
        self.dropout = nn.Dropout(self.dropout_keep_prob)
        self.logits = nn.Linear(512*32, self.num_classes)
        torch.nn.init.xavier_uniform(self.logits.weight)
        torch.nn.init.constant(self.logits.bias, 0.1)
        self.softmax_cls = nn.Softmax()
        self.softmax_img = nn.Softmax()

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
        feature_keyframe, pred_keyframes = self.resnet_feature(x_keyframes)   # [8 512 7 7] [8 400]
        pred_keyframes = self.softmax_cls(pred_keyframes)
        #pred_keyframes = torch.unsqueeze(pred_keyframes, dim=0)

        # slice feature
        feat_slices = torch.split(feature_keyframe, dim=0, split_size_or_sections=1)  # 8*[1 512 7 7]

        ###############################################################
        # processing for 3D conv
        # compute optical flow
        flow_data_key1 = torch.squeeze(
            torch.cat((data_key1, data_key1, data_key1, data_key1, data_key1, data_key1, data_key1), dim=2))  # [3 7 224 224]
        flow_data1 = torch.cat((flow_data_key1, torch.squeeze(nokey1)), dim=0).permute(1, 0, 2, 3)  # [7 6 224 224]
        flow_data_key2 = torch.squeeze(
            torch.cat((data_key2, data_key2, data_key2, data_key2, data_key2, data_key2, data_key2), dim=2))
        flow_data2 = torch.cat((flow_data_key2, torch.squeeze(nokey2)), dim=0).permute(1, 0, 2, 3)  # [7 6 224 224]
        flow_data_key3 = torch.squeeze(
            torch.cat((data_key3, data_key3, data_key3, data_key3, data_key3, data_key3, data_key3), dim=2))
        flow_data3 = torch.cat((flow_data_key3, torch.squeeze(nokey3)), dim=0).permute(1, 0, 2, 3)  # [7 6 224 224]
        flow_data_key4 = torch.squeeze(
            torch.cat((data_key4, data_key4, data_key4, data_key4, data_key4, data_key4, data_key4), dim=2))
        flow_data4 = torch.cat((flow_data_key4, torch.squeeze(nokey4)), dim=0).permute(1, 0, 2, 3)  # [7 6 224 224]
        flow_data_key5 = torch.squeeze(
            torch.cat((data_key5, data_key5, data_key5, data_key5, data_key5, data_key5, data_key5), dim=2))
        flow_data5 = torch.cat((flow_data_key5, torch.squeeze(nokey5)), dim=0).permute(1, 0, 2, 3)  # [7 6 224 224]
        flow_data_key6 = torch.squeeze(
            torch.cat((data_key6, data_key6, data_key6, data_key6, data_key6, data_key6, data_key6), dim=2))
        flow_data6 = torch.cat((flow_data_key6, torch.squeeze(nokey6)), dim=0).permute(1, 0, 2, 3)  # [7 6 224 224]
        flow_data_key7 = torch.squeeze(
            torch.cat((data_key7, data_key7, data_key7, data_key7, data_key7, data_key7, data_key7), dim=2))
        flow_data7 = torch.cat((flow_data_key7, torch.squeeze(nokey7)), dim=0).permute(1, 0, 2, 3)  # [7 6 224 224]
        flow_data_key8 = torch.squeeze(
            torch.cat((data_key8, data_key8, data_key8, data_key8, data_key8, data_key8, data_key8), dim=2))
        flow_data8 = torch.cat((flow_data_key8, torch.squeeze(nokey8)), dim=0).permute(1, 0, 2, 3)  # [7 6 224 224]

        # flownet
        concat_flow_data = torch.cat((flow_data1, flow_data2, flow_data3, flow_data4,
                                            flow_data5, flow_data6, flow_data7, flow_data8), dim=0)   # [56 6 224 224]
        concat_flow_data_resize = self.flownetresize(concat_flow_data)  # [56 6 28 28]
        flow = self.flownets(concat_flow_data_resize)  # [56 2 7 7]


        # flow slice
        flow_slices = torch.chunk(flow, dim=0, chunks=num_mini_clips)  # 8 * [7 2 7 7]

        # warping
        warp_conv1 = self.warping_function(flow_slices[0], feat_slices[0], lenght_mini_clip)
        warp_conv2 = self.warping_function(flow_slices[1], feat_slices[1], lenght_mini_clip)
        warp_conv3 = self.warping_function(flow_slices[2], feat_slices[2], lenght_mini_clip)
        warp_conv4 = self.warping_function(flow_slices[3], feat_slices[3], lenght_mini_clip)
        warp_conv5 = self.warping_function(flow_slices[4], feat_slices[4], lenght_mini_clip)
        warp_conv6 = self.warping_function(flow_slices[5], feat_slices[5], lenght_mini_clip)
        warp_conv7 = self.warping_function(flow_slices[6], feat_slices[6], lenght_mini_clip)
        warp_conv8 = self.warping_function(flow_slices[7], feat_slices[7], lenght_mini_clip)

        concat_feat = torch.cat((feat_slices[0], warp_conv1,
                                       feat_slices[1], warp_conv2,
                                       feat_slices[2], warp_conv3,
                                       feat_slices[3], warp_conv4,
                                       feat_slices[4], warp_conv5,
                                       feat_slices[5], warp_conv6,
                                       feat_slices[6], warp_conv7,
                                       feat_slices[7], warp_conv8),
                                       dim=0)  # [64 512 7 7]

        feat_re = torch.unsqueeze(concat_feat, dim=0)  # [1 64 512 7 7]
        feat_t  = feat_re.permute(0, 2, 1, 3, 4)

        # 3D inception
        # mixed_4b
        feature = self.inception_3D_1(feat_t)
        feature = self.inception_3D_2(feature)
        feature = self.inception_3D_3(feature)

        # avg pool
        feat_avg = self.avg_pool(feature)

        # dropout
        feat_dropout = self.dropout(feat_avg)

        # flatten
        feat_flat = torch.flatten(feat_dropout)

        # prediction
        pred_video = self.logits(feat_flat)

        # loss for video
        pred_video = self.softmax_cls(pred_video)
        pred_video = torch.unsqueeze(pred_video, dim=0)

        return pred_keyframes, pred_video

    def warping_function(self, flow, feat_cam, duration):

        feat_keys = torch.cat((feat_cam, feat_cam, feat_cam, feat_cam, feat_cam, feat_cam, feat_cam), dim=0)

        return warp(feat_keys, flow)  # [7 512 7 7]


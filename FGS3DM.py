import torch.nn as nn
import torch

from models.resnet import resnet101
from models.FlowNetS import flownets
from models.inception3D import InceptionModule, Unit3D
from models.warping import warp


class FGS3DM(nn.Module):

    def __init__(self, num_classes=400, num_frames=64, num_keyframe=8, dropout_keep_prob=0.5):
        super(FGS3DM, self).__init__()

        self.num_frames = num_frames
        self.num_keyframe = num_keyframe
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob

        self.resnet_feature = resnet101(pretrained=True)
        num_ftrs = self.resnet_feature.fc.in_features
        self.resnet_feature.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # x: [mini_batch_size 3 64 112 112]

        mini_batch_size = x.shape[0]  # mini_batch_size = batchisze/n_gpu
        n_channel = x.shape[1]
        n_frames = x.shape[2]
        assert(n_frames == self.num_frames)
        n_hight = x.shape[3]
        n_width = x.shape[4]
        num_mini_clips = int(self.num_keyframe)
        lenght_mini_clip = int(self.num_frames / self.num_keyframe)

        ###############################################################
        # data preparing
        # slice key frames
        x_trunk = torch.split(x, 1, dim=2)  # x_trunk: 64 * [mb 3 1 224 224]

        """
        # key frames
        data_key1 = x_trunk[0]                        # data_key1: [mb 3 1 224 224]
        data_key2 = x_trunk[0 + lenght_mini_clip * 1] # data_key2: [mb 3 1 224 224]
        data_key3 = x_trunk[0 + lenght_mini_clip * 2]
        data_key4 = x_trunk[0 + lenght_mini_clip * 3]
        data_key5 = x_trunk[0 + lenght_mini_clip * 4]
        data_key6 = x_trunk[0 + lenght_mini_clip * 5]
        data_key7 = x_trunk[0 + lenght_mini_clip * 6]
        data_key8 = x_trunk[0 + lenght_mini_clip * 7]

        # No key frames
        nokey1 = torch.cat(x_trunk[1:lenght_mini_clip * 1], dim=2)                         # nokey1: [mb 3 7 224 224]
        nokey2 = torch.cat(x_trunk[1 + lenght_mini_clip * 1:lenght_mini_clip * 2], dim=2)  # nokey1: [mb 3 7 224 224]
        nokey3 = torch.cat(x_trunk[1 + lenght_mini_clip * 2:lenght_mini_clip * 3], dim=2)
        nokey4 = torch.cat(x_trunk[1 + lenght_mini_clip * 3:lenght_mini_clip * 4], dim=2)
        nokey5 = torch.cat(x_trunk[1 + lenght_mini_clip * 4:lenght_mini_clip * 5], dim=2)
        nokey6 = torch.cat(x_trunk[1 + lenght_mini_clip * 5:lenght_mini_clip * 6], dim=2)
        nokey7 = torch.cat(x_trunk[1 + lenght_mini_clip * 6:lenght_mini_clip * 7], dim=2)
        nokey8 = torch.cat(x_trunk[1 + lenght_mini_clip * 7:], dim=2)
        """

        ###############################################################
        # processing for key frames
        # concat key frames
        x_keyframes = torch.cat(x_trunk[0::lenght_mini_clip], dim=2)  # [mb 3 8 224 224]

        # reshape [8 3 224 224]
        x_keyframes = x_keyframes.permute(0, 2, 1, 3, 4)  # [mb 8 3 224 224]
        x_keyframes = torch.reshape(x_keyframes, (mini_batch_size*lenght_mini_clip, n_channel, n_hight, n_width))  # [mb*8 3 224 224]

        # extract features for key frames
        feature_keyframe, pred_keyframes = self.resnet_feature(x_keyframes)   # [mb*8 1024 14 14] [mb*8 400]
        #pred_keyframes = self.softmax_cls(pred_keyframes)

        """
        ###############################################################
        # processing for 3D conv
        # compute optical flow
        # slice feature
        feat_slices = torch.split(feature_keyframe, dim=0, split_size_or_sections=1)  # 8*[1 1024 14 14]
        
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
        concat_flow_data_resize = self.flownetresize(concat_flow_data)  # [56 6 56 56]
        # flow_big = self.flownets(concat_flow_data)  # [56 2 56 56]
        flow = self.flownets(concat_flow_data_resize)  # [56 2 14 14]


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
                                       dim=0)  # [64 1024 14 14]

        feat_re = torch.unsqueeze(concat_feat, dim=0)  # [1 64 1024 14 14]
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
        """
        return pred_keyframes

    def warping_function(self, flow, feat_cam, duration):

        feat_keys = torch.cat((feat_cam, feat_cam, feat_cam, feat_cam, feat_cam, feat_cam, feat_cam), dim=0)

        return warp(feat_keys, flow)  # [7 512 7 7]


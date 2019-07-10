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

class FGS3DFLOW(nn.Module):

    def __init__(self, num_classes=400, num_frames=64, num_keyframe=8, dropout_keep_prob=0.5):
        super(FGS3DFLOW, self).__init__()

        self.num_frames = num_frames
        self.num_keyframe = num_keyframe
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob

        ##############################################
        # Load flownet
        ##############################################
        
        self.flownetresize = nn.AvgPool2d(kernel_size=4, stride=4)
        FlowNet_state_dict = torch.load('/home/weik/pretrainedmodels/FlowNetS/flownets_from_caffe.pth.tar.pth')
        self.flownets = flownets(FlowNet_state_dict)
        set_parameter_requires_grad(self.flownets)
        self.flownets = flownets()


        # self.inception_3D_1 = InceptionModule(1024, [112, 144, 288, 32, 64, 64], 'mixed_4f', )
        self.inception_3D_flow_1 = InceptionModule(2, [256,160,320,32,128,128], 'mixed_4f')
        self.inception_3D_flow_2 = InceptionModule(256+320+128+128, [256,160,320,32,128,128], 'mixed_5b')
        self.inception_3D_flow_3 = InceptionModule(256+320+128+128, [384,192,384,48,128,128], 'mixed_5c')

        self.avg_pool_flow = nn.AvgPool3d(kernel_size=[2, 14, 14], stride=(2, 1, 1))
        self.dropout_flow = nn.Dropout(self.dropout_keep_prob)
        self.logits_flow = nn.Linear((384+384+128+128)*28, self.num_classes)
        torch.nn.init.normal_(self.logits_flow.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.logits_flow.bias, 0.0)

        set_parameter_requires_grad(self.flownets)


    def forward(self, x):

        # x: [batchisze/n_gpu 3 64 112 112]

        num_mini_clips = int(self.num_keyframe)
        lenght_mini_clip = int(self.num_frames / self.num_keyframe)

        ###############################################################
        # data preparing
        # slice key frames
        x_trunk = torch.split(x, 1, dim=2)  # x_trunk: 64 * [1 3 1 224 224]

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
        self.flownets.eval()
        flow = self.flownets(concat_flow_data_resize)  # [56 2 14 14]


        feat_re = torch.unsqueeze(flow, dim=0)  # [1 56 2 14 14]
        feat_t  = feat_re.permute(0, 2, 1, 3, 4)

        # 3D inception
        # mixed_4b
        feature = self.inception_3D_flow_1(feat_t)
        feature = self.inception_3D_flow_2(feature)
        feature = self.inception_3D_flow_3(feature)

        # avg pool
        feat_avg = self.avg_pool_flow(feature)

        # dropout
        feat_dropout = self.dropout_flow(feat_avg)

        # flatten
        feat_flat = torch.flatten(feat_dropout)

        # prediction
        pred_video = self.logits_flow(feat_flat)

        # # loss for video
        # pred_video = self.softmax_cls(pred_video)
        pred_video = torch.unsqueeze(pred_video, dim=0)

        return pred_video



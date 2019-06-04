import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from kinetics_img_dataset import KineticsImg
from utils import Logger
from train import train_epoch
from validation import val_epoch
from FGS3D_img import FGS3DIMG


def get_fine_tuning_parameters(model, fixed_names=None):
    if fixed_names is None:
        return model.parameters()

    parameters = []
    for k, v in model.named_parameters():
        fixed_flag = False
        for fixed_name in fixed_names:
            if fixed_name in k:
                fixed_flag = True
                break
        if fixed_flag:
            v.requires_grad = False
        else:
            v.requires_grad = True

        parameters.append({'params': v})

    return parameters


if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    modality = 'RGB'
    num_class = opt.n_classes
    if opt.arch == 'FGS3D-0':
        input_channel = 3 if modality == 'RGB' else 2

        model = FGS3DIMG(num_classes=num_class, dropout_keep_prob=0.0)
        model = torch.nn.DataParallel(model).cuda()
        parameters = get_fine_tuning_parameters(model, ['flownets', 'resnet_feature.conv1', 'resnet_feature.bn',
                                                        'resnet_feature.layer1', 'resnet_feature.layer2',
                                                        'resnet_feature.layer3', 'resnet_feature.layer4',
                                                        'resnet_feature.layer5', 'resnet_feature.fc'])
    else:
        IOError(opt.arch)

    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value),
            norm_method
        ])

        training_data = KineticsImg('/data/Kinetics400/trainvalList_test/train_imglist_all_s30.cvs', spatial_transform=spatial_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'acc_img', 'lr', 'epoch_time'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'acc_img', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=30)
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        validation_data = KineticsImg('/data/Kinetics400/trainvalList_test/val_imglist_all_s30.cvs', spatial_transform=spatial_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

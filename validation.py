import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:

            targets_vid = targets.cuda(non_blocking=True)
            targets_img = torch.cat((targets, targets, targets, targets, targets, targets, targets, targets), dim=0)
            targets_img = targets_img.cuda(non_blocking=True)
        inputs = Variable(inputs)
        targets_vid = Variable(targets_vid)
        targets_img = Variable(targets_img)

        outputs = model(inputs)
        loss_vid = criterion(outputs[1], targets_vid)
        loss_img = criterion(outputs[0], targets_img)
        loss = loss_vid + loss_img
        acc = calculate_accuracy(outputs[1], targets_vid)

        losses.update(loss.data.cpu(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg

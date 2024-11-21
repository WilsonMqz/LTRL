import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from utils.utils_algo import AverageMeter, accuracy


def train(train_loader, model, criterion, optimizer, epoch):
    epoch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        inputs = inputs.cuda()
        target = target.cuda()
        output = model(inputs)
        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_time.update(time.time() - end)

    return epoch_time.avg, losses.avg, top1.avg


def validate(val_loader, model, criterion, epoch, num_classes, args):
    epoch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    medium1 = args.medium1
    medium2 = args.medium2

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            inputs = inputs.cuda()
            target = target.cuda()

            output = model(inputs)
            loss = criterion(output, target)

            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        epoch_time.update(time.time() - end)

        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        # out_cls_acc = 'Epoch [%s] Class Accuracy: %s' % (
        #     str(epoch), (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        # print(out_cls_acc)
        many_cls = cls_acc[:medium1]
        medium_cls = cls_acc[medium1:medium2]
        few_cls = cls_acc[medium2:]
        many_acc = many_cls.sum() / (len(many_cls)) * 100
        medium_acc = medium_cls.sum() / (len(medium_cls)) * 100
        few_acc = few_cls.sum() / (len(few_cls)) * 100
    return losses.avg, top1.avg, many_acc, medium_acc, few_acc

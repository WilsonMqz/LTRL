import argparse
import copy
import os

import numpy as np
import torch
from torch import nn

import SL
import TL
from models.wideresnet import WideResNet
from utils.utils_algo import adjust_learning_rate
from utils.utils_dataloaders import im_cifar100_loader, im_cifar10_loader

parser = argparse.ArgumentParser(description='Revisiting Consistency Regularization for Deep Partial Label Learning')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--dataset', type=str, default='tiny-imagenet',
                    choices=['cifar100', 'cifar10', 'svhn', 'stl10', 'tiny-imagenet'])
parser.add_argument('--model', type=str, choices=['widenet', 'lenet'], default='widenet')
parser.add_argument('--lr', default=1e-1, type=float, help='learning-rate')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=2e-4, type=float, help='weight-decay')
parser.add_argument('--extend-size', default=30, type=int, help='extend size of stochastic label')
parser.add_argument('--fix-size', default=50, type=int, help='fixed size of stochastic label')
parser.add_argument('--seed', default=100, type=int, help='fix random number for data sampling')
parser.add_argument('--alpha', default=0, type=int, help='alpha value for mix_up')
parser.add_argument('--data-dir', default='./data/', type=str)
parser.add_argument('--method', default='SL', choices=['TL', 'SL'])
parser.add_argument('--imb-type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb-factor', default=0.02, type=float, help='imbalance factor')

args = parser.parse_args()

# fix random seed
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

save_dir = "./results" + '_' + str(args.seed) + "/res"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = "res_ds_{}_mo_{}_me_{}_lr_{}_wd_{}_e_{}_bs_{}_imt_{}_imf_{}_fix_{}_ex_{}_alpha_{}.csv" \
    .format(args.dataset, args.model, args.method, args.lr, args.wd, args.epochs, args.batch_size,
            args.imb_type, args.imb_factor, args.fix_size, args.extend_size, args.alpha)
model_save_dir = "./results" + '_' + str(args.seed) + "/best_model"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
save_model_name = "best_model_ds_{}_mo_{}_me_{}_lr_{}_wd_{}_e_{}_bs_{}_imt_{}_imf_{}_fix_{}_ex_{}_alpha_{}.csv" \
    .format(args.dataset, args.model, args.method, args.lr, args.wd, args.epochs, args.batch_size,
            args.imb_type, args.imb_factor, args.fix_size, args.extend_size, args.alpha)
model_save_path = os.path.join(model_save_dir, save_model_name)
record_save_path = os.path.join(save_dir, save_name)
with open(record_save_path, 'a') as f:
    f.writelines("epoch,train_loss,train_acc,val_loss,val_acc,many_acc,medium_acc,few_acc\n")


def main():
    if args.dataset == 'cifar100':
        num_classes = 100
        args.medium1 = 35  # 0~35: >= 100; 36~70: <100 & >=20; 71~100: <20
        args.medium2 = 70
    elif args.dataset in ['cifar10', 'svhn', 'stl10']:
        if args.dataset == 'stl10':
            args.medium1 = 4
            args.medium2 = 7
        else:
            args.medium1 = 5
            args.medium2 = 8
        num_classes = 10
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
        args.medium1 = 82
        args.medium2 = 164

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    if args.model == 'widenet':
        model = WideResNet(34, num_classes, widen_factor=10, dropRate=0.0)
    else:
        assert "Unknown model"
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)

    best_acc = 0

    if args.dataset == 'cifar100':
        train_loader, test_loader, cls_num_list = im_cifar100_loader(args)
    elif args.dataset == 'cifar10':
        train_loader, test_loader, cls_num_list = im_cifar10_loader(args)

    print('cls num list:')
    print(cls_num_list)

    for epoch in range(args.epochs):

        # fine tune after mixup
        if epoch >= 190:
            args.alpha = 0

        adjust_learning_rate(optimizer, epoch, args)

        criterion = nn.CrossEntropyLoss().to(device)

        if args.method == 'TL':
            train_time, train_loss, trian_acc = TL.train(train_loader, model, criterion, optimizer, epoch)
            test_loss, test_acc, many_acc, medium_acc, few_acc = TL.validate(test_loader, model, criterion, epoch,
                                                                             num_classes, args)
        elif args.method == 'SL':
            criterion = nn.CrossEntropyLoss().to(device)
            train_time, train_loss, trian_acc = SL.train(train_loader, model, num_classes, optimizer, device,
                                                         args.alpha)
            test_loss, test_acc, many_acc, medium_acc, few_acc = SL.validate(test_loader, model, criterion, epoch,
                                                                             num_classes, args)
        print("Epoch: [{}], lr: {:.6f}, Time: {:.3f}, Tr_loss: {:.4f}, Tr_acc: {:.3f}, Te_loss: {:.4f}, "
              "Te_acc: {:.3f}, Many_acc: {:.3f}, Medium_acc: {:.3f}, Few_acc: {:.3f}"
              .format(epoch + 1, optimizer.param_groups[-1]['lr'], train_time, train_loss, trian_acc, test_loss,
                      test_acc, many_acc, medium_acc, few_acc))
        with open(record_save_path, 'a') as f:
            f.writelines("{},{:.4f},{:.3f},{:.4f},{:.3f},{:.3f},{:.3f},{:.3f}\n"
                         .format(epoch, train_loss, trian_acc, test_loss, test_acc, many_acc, medium_acc, few_acc))

        if best_acc < test_acc:
            best_acc = test_acc
            if args.method == 'SL':
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, model_save_path)

    print("best accuracy is %.2f\n" % best_acc)
    with open(record_save_path, 'a') as f:
        f.writelines("max,{:.2f}\n".format(best_acc))


if __name__ == '__main__':
    main()

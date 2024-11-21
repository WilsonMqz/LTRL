import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import imbalance_cifar_TL, imbalance_cifar_SL


def im_cifar100_loader(args):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if args.method == 'TL':
        train_dataset = imbalance_cifar_TL.Im_CIFAR100(root=args.data_dir, imb_type=args.imb_type,
                                                       imb_factor=args.imb_factor, seed=args.seed,
                                                       train=True, download=True, transform=transform_train)
    elif args.method == 'SL':
        train_dataset = imbalance_cifar_SL.Im_CIFAR100(root=args.data_dir, imb_type=args.imb_type,
                                                       imb_factor=args.imb_factor, seed=args.seed,
                                                       extend_size=args.extend_size, fix_size=args.fix_size,
                                                       train=True, download=True, transform=transform_train)

    test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader, train_dataset.get_cls_num_list()


def im_cifar10_loader(args):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if args.method == 'TL':
        train_dataset = imbalance_cifar_TL.Im_CIFAR10(root=args.data_dir, imb_type=args.imb_type,
                                                      imb_factor=args.imb_factor, seed=args.seed,
                                                      train=True, download=True, transform=transform_train)
    elif args.method == 'SL':
        train_dataset = imbalance_cifar_SL.Im_CIFAR10(root=args.data_dir, imb_type=args.imb_type,
                                                      imb_factor=args.imb_factor, seed=args.seed,
                                                      extend_size=args.extend_size, fix_size=args.fix_size,
                                                      train=True, download=True, transform=transform_train)

    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader, train_dataset.get_cls_num_list()

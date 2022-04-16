import json
import os

import torch
from torchvision import transforms, datasets

from aip.models import lenet5, alexnet, vgg16, vgg16_bn, resnet18, resnet34, resnet50, resnet101, resnet152, \
    resnext50_32x4d, \
    resnext101_32x8d

__all__ = ["get_dataset", "get_transform", "get_arch_net"]


def get_transform(dataset_name, train=True):
    assert dataset_name in ('cifar10', 'flower_data'), f'dataset {dataset_name} is not supported.'

    data_transform = None
    if dataset_name == 'cifar10':
        data_transform = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),

            'val': transforms.Compose([
                transforms.Resize(36),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    if dataset_name == 'flower_data':
        data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    if train:
        return data_transform['train']
    else:
        return data_transform['val']


def _load_cifar10(dataset_root, dataset_name, batch_size, nw):
    train_dataset = datasets.CIFAR10(dataset_root, train=True, download=True,
                                     transform=get_transform(dataset_name, train=True))
    valid_dataset = datasets.CIFAR10(dataset_root, train=False, download=True,
                                     transform=get_transform(dataset_name, train=True))
    print(f'using {len(train_dataset)} images for training, {len(valid_dataset)} images for validation.')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    return train_dataset, valid_dataset, train_loader, valid_loader


def _load_flower(dataset_root, dataset_name, batch_size, nw):
    train_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'train'),
                                         transform=get_transform(dataset_name, train=True))
    valid_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'val'),
                                         transform=get_transform(dataset_name, train=False))
    print(f'using {len(train_dataset)} images for training, {len(valid_dataset)} images for validation.')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    return train_dataset, valid_dataset, train_loader, valid_loader


def get_dataset(root, dataset_name, batch_size, num_workers=0):
    dataset_root = os.path.join(root, 'dataset', dataset_name)
    assert os.path.exists(dataset_root), f'{dataset_root} path does not exist.'

    assert dataset_name in ('cifar10', 'flower_data'), f'dataset {dataset_name} is not supported.'
    train_dataset, valid_dataset, train_loader, valid_loader = None, None, None, None

    if dataset_name == 'cifar10':
        train_dataset, valid_dataset, train_loader, valid_loader = _load_cifar10(dataset_root, dataset_name, batch_size,
                                                                                 num_workers)

    if dataset_name == 'flower_data':
        train_dataset, valid_dataset, train_loader, valid_loader = _load_flower(dataset_root, dataset_name, batch_size,
                                                                                num_workers)

    return dataset_root, (train_dataset, valid_dataset), (train_loader, valid_loader)


def get_arch_net(root, arch, train_dataset, train=True, **kwargs):
    model_root = os.path.join(root, 'saved', arch)
    assert os.path.exists(model_root), f'{model_root} path does not exist.'

    assert arch in (
        'lenet5', 'alexnet', 'vgg16', 'vgg16_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d',
        'resnext101_32x8d'), f'{arch} is not supported.'

    if train:
        class_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in class_list.items())
        json_str = json.dumps(cla_dict, indent=4)
        with open(os.path.join(model_root, 'class_indices.json'), 'w') as json_file:
            json_file.write(json_str)
    else:
        json_path = os.path.join(model_root, 'class_indices.json')
        assert os.path.exists(json_path), f'file: {json_path} dose not exist.'
        cla_dict = json.load(open(json_path, 'r'))

    net = None
    if arch == 'lenet5':
        net = lenet5(**kwargs)
    elif arch == 'alexnet':
        net = alexnet(**kwargs)
    elif arch == 'vgg16':
        net = vgg16(**kwargs)
    elif arch == 'vgg16_bn':
        net = vgg16_bn(**kwargs)
    elif arch == 'resnet18':
        net = resnet18(**kwargs)
    elif arch == 'resnet34':
        net = resnet34(**kwargs)
    elif arch == 'resnet50':
        net = resnet50(**kwargs)
        # import torchvision
        # net = torchvision.models.resnet50(pretrained=True)
    elif arch == 'resnet101':
        net = resnet101(**kwargs)
    elif arch == 'resnet152':
        net = resnet152(**kwargs)
    elif arch == 'resnext50_32x4d':
        net = resnext50_32x4d(**kwargs)
    elif arch == 'resnext101_32x8d':
        net = resnext101_32x8d(**kwargs)

    return model_root, net, cla_dict

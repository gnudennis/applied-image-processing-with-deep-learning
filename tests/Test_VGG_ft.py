import argparse
import os

import torch
from PIL import Image
from torch import nn, optim
from util import get_dataset, get_arch_net, get_transform

from aip import train_wrapper, predict_wrapper
from aip.utils import get_dataloader_workers, try_all_gpus, show_images


def train_args():
    parser = argparse.ArgumentParser(description="train and save your model")
    parser.add_argument('--arch', default='vgg16_bn', help='architecture')
    parser.add_argument('--train-mode', default='ft', help='choose train mode (start/ft)')
    parser.add_argument('--weights-pth', default='vgg16_bn-6c64b313.pth', help='net weigths path only used in ft')
    parser.add_argument('--dataset', default='flower_data', help='choose your dataset')
    parser.add_argument('--saved-pth', default='vgg16_bn_ft.pth', help='path for the trained model')

    parser.add_argument('--param_group', default=False, help="param group")
    parser.add_argument('--batch-size', default=64, help='batch size')
    parser.add_argument('--learning-rate', default=5e-4, help='learning rate')
    parser.add_argument('--num-epochs', default=5, help="number of epochs")

    return parser


def test_args():
    parser = argparse.ArgumentParser(description="test model")
    parser.add_argument('--arch', default='vgg16_bn', help='architecture')
    parser.add_argument('--weights-pth', default='vgg16_bn_ft.pth', help='net weigths path')
    parser.add_argument('--dataset', default='flower_data', help='choose your dataset')
    parser.add_argument('--num-classes', default=5, help='num of classes')

    return parser


def train(args: argparse.Namespace):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get project root

    # load dataset
    dataset_root, (train_dataset, valid_dataset), (train_loader, valid_loader) = get_dataset(
        root, args.dataset, args.batch_size,
        num_workers=get_dataloader_workers(args.batch_size)
    )

    # load
    model_root, net, cla_dict = get_arch_net(root, args.arch, train_dataset)

    if args.train_mode == 'ft':
        model_weight_path = os.path.join(model_root, args.weights_pth)
        assert os.path.exists(model_weight_path), f'file {model_root} does not exist.'
        net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
        # for param in net.parameters():
        #     param.requires_grad = False

        in_channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_channel, len(cla_dict))

    if args.param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if param.requires_grad and name not in ["fc.weight", "fc.bias"]]
        optimizer = optim.Adam([{'params': params_1x},
                                {'params': net.fc.parameters(), 'lr': args.learning_rate * 10}],
                               lr=args.learning_rate, weight_decay=0.001)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate,
                               weight_decay=0.001)
    loss = nn.CrossEntropyLoss(reduction='none')

    devices = try_all_gpus()
    saved_path = os.path.join(model_root, args.saved_pth)
    train_wrapper(net, train_loader, valid_loader, loss, optimizer, args.num_epochs, devices, saved_path)

    from matplotlib import pyplot as plt
    plt.show()


def predict(args: argparse.Namespace, image_name: str):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get project root

    dataset_root = os.path.join(root, 'dataset', args.dataset)
    assert os.path.exists(dataset_root), f'{dataset_root} path does not exist.'

    model_root, net, cla_dict = get_arch_net(root, args.arch, None, train=False,
                                             num_classes=args.num_classes)
    model_weight_path = os.path.join(model_root, args.weights_pth)
    assert os.path.exists(model_weight_path), f'file {args.weights_pth} does not exist.'
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    # print('[missing_keys]:', *missing_keys, sep='\n')
    # print('[unexpected_keys]:', *unexpected_keys, sep='\n')

    test_image_path = os.path.join(dataset_root, 'pred', image_name)
    assert os.path.exists(test_image_path), f'file: {test_image_path} dose not exist.'

    transform = get_transform(args.dataset, train=False)
    pimg = Image.open(test_image_path)
    # plt.imshow(pimg)
    # [N, C, H, W]
    img = torch.unsqueeze(transform(pimg), dim=0)
    probs, classes = predict_wrapper(net, img)

    titles = [f'{os.path.splitext(os.path.basename(test_image_path))[0]}\n'
              f'({cla_dict[str(int(classes))]}:{float(probs):.3f})']
    show_images([pimg], 1, 1, titles, scale=2.5)


def batch_predict(args: argparse.Namespace,
                  batch_size: int = 3,
                  shows: int = 12,
                  formats: str = ['.jpg', '.jpeg', '.webp']):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get project root

    dataset_root = os.path.join(root, 'dataset', args.dataset)
    assert os.path.exists(dataset_root), f'{dataset_root} path does not exist.'

    model_root, net, cla_dict = get_arch_net(root, args.arch, None, train=False,
                                             num_classes=args.num_classes)
    model_weight_path = os.path.join(model_root, args.weights_pth)
    assert os.path.exists(model_weight_path), f'file {args.weights_pth} does not exist.'
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    # print('[missing_keys]:', *missing_keys, sep='\n')
    # print('[unexpected_keys]:', *unexpected_keys, sep='\n')

    transform = get_transform(args.dataset, train=False)

    image_pred_root = os.path.join(dataset_root, 'pred')
    assert os.path.exists(image_pred_root), f'file: {image_pred_root} dose not exist.'

    img_path_list = [os.path.join(image_pred_root, item) for item in os.listdir(image_pred_root)
                     if any([item.endswith(fmt) for fmt in formats])]

    probs = []
    classes = []
    pil_img_list = []
    for ids in range(0, len(img_path_list) // batch_size + 1):
        pil_img_batch = []
        img_batch = []
        for each_img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
            assert os.path.exists(each_img_path), f'file: {each_img_path} dose not exist.'
            pil_img = Image.open(each_img_path)
            pil_img_batch.append(pil_img)
            img_batch.append(transform(pil_img))

        if img_batch:
            batch_img = torch.stack(img_batch, dim=0)
            probs_batch, classes_batch = predict_wrapper(net, batch_img)

            pil_img_list.extend(pil_img_batch)
            probs.extend(probs_batch.numpy().tolist())
            classes.extend(classes_batch.numpy().tolist())

    for idx, (img_path, pro, cla) in enumerate(zip(img_path_list, probs, classes)):
        print(f'{idx + 1:<3}{os.path.basename(img_path):<20}'
              f'class: {cla_dict[str(cla)]:<20}prob: {pro:.3f}')

    titles = [f'{os.path.splitext(os.path.basename(img_path))[0]}\n'
              f'({cla_dict[str(cla)]}:{pro:.3f})'
              for (img_path, pro, cla) in zip(img_path_list, probs, classes)]
    shows = min([shows, len(pil_img_list)])
    show_images(pil_img_list[:(shows // 3) * 3], shows // 3, 3, titles[:(shows // 3) * 3], scale=2.5)


if __name__ == '__main__':
    parser = train_args()
    args, unknown = parser.parse_known_args()
    train(args)

    # parser = test_args()
    # args, unknown = parser.parse_known_args()
    # # predict(args, 'sunflowers2.webp')
    # batch_predict(args, batch_size=5, shows=15)

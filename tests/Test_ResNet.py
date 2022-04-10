import json
import os

import torch
from PIL import Image
from torch import nn, optim
from torchvision import transforms, datasets

from aip import train, predict as aip_predict
from aip.models import resnet34
from aip.utils import get_dataloader_workers, try_all_gpus, show_images


def train_ft(batch_size=64, num_epochs=5, learning_rate=0.0001, param_group=True):
    """训练脚本(finetune)"""
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get data root path
    image_path = os.path.join(root, 'dataset', 'flower_data')  # flower data set path
    assert os.path.exists(image_path), f'{image_path} path does not exist.'
    model_path = os.path.join(root, 'saved', 'resnet')
    assert os.path.exists(model_path), f'{model_path} path does not exist.'

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
    train_dataset = datasets.ImageFolder(os.path.join(image_path, 'train'), transform=data_transform['train'])
    valid_dataset = datasets.ImageFolder(os.path.join(image_path, 'val'), transform=data_transform['val'])
    print(f'using {len(train_dataset)} images for training, {len(valid_dataset)} images for validation.')

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(model_path, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)

    nw = get_dataloader_workers(batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    # import torchvision
    # net = torchvision.models.resnet50(pretrained=True)

    # load pretrain weights
    net = resnet34()
    model_weight_path = os.path.join(model_path, 'resnet34-pre.pth')
    assert os.path.exists(model_weight_path), f'file {model_path} does not exist.'
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, len(flower_list))
    loss = nn.CrossEntropyLoss(reduction='none')

    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if param.requires_grad and name not in ["fc.weight", "fc.bias"]]
        optimizer = optim.Adam([{'params': params_1x},
                                {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
                               lr=learning_rate, weight_decay=0.001)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                               weight_decay=0.001)

    devices = try_all_gpus()
    saved_path = os.path.join(model_path, 'resnet34_ft.pth')
    train(net, train_loader, valid_loader, loss, optimizer, num_epochs, devices, saved_path)


def predict(image_name):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get data root path
    image_path = os.path.join(root, 'dataset', 'flower_data')  # flower data set path
    assert os.path.exists(image_path), f'{image_path} path does not exist.'
    model_path = os.path.join(root, 'saved', 'resnet')
    assert os.path.exists(model_path), f'{model_path} path does not exist.'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    net = resnet34(num_classes=5)
    model_weight_path = os.path.join(model_path, 'resnet34_ft.pth')
    assert os.path.exists(model_weight_path), f'file {model_path} does not exist.'
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    # print('[missing_keys]:', *missing_keys, sep='\n')
    # print('[unexpected_keys]:', *unexpected_keys, sep='\n')
    net = net.to(device)

    # read cla_dict
    json_path = os.path.join(model_path, 'class_indices.json')
    assert os.path.exists(json_path), f'file: {json_path} dose not exist.'
    cla_dict = json.load(open(json_path, 'r'))

    test_image_path = os.path.join(image_path, 'pred', image_name)
    assert os.path.exists(test_image_path), f'file: {test_image_path} dose not exist.'

    pimg = Image.open(test_image_path)
    # plt.imshow(pimg)
    # [N, C, H, W]
    img = torch.unsqueeze(data_transform(pimg), dim=0)
    probs, classes = aip_predict(net, img)

    titles = [f'{os.path.splitext(os.path.basename(test_image_path))[0]}\n'
              f'({cla_dict[str(int(classes))]}:{float(probs):.3f})']
    show_images([pimg], 1, 1, titles, scale=2.5)


def batch_predict(batch_size=8, formats=['.jpg', '.jpeg', '.webp'], shows=12):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get data root path
    image_path = os.path.join(root, 'dataset', 'flower_data')  # flower data set path
    assert os.path.exists(image_path), f'{image_path} path does not exist.'
    model_path = os.path.join(root, 'saved', 'resnet')
    assert os.path.exists(model_path), f'{model_path} path does not exist.'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    net = resnet34(num_classes=5)
    model_weight_path = os.path.join(model_path, 'resnet34_ft.pth')
    assert os.path.exists(model_weight_path), f'file {model_path} does not exist.'
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net = net.to(device)

    # read cla_dict
    json_path = os.path.join(model_path, 'class_indices.json')
    assert os.path.exists(json_path), f'file: {json_path} dose not exist.'
    cla_dict = json.load(open(json_path, 'r'))

    image_pred_root = os.path.join(image_path, 'pred')
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
            img_batch.append(data_transform(pil_img))

        if img_batch:
            batch_img = torch.stack(img_batch, dim=0)
            probs_batch, classes_batch = aip_predict(net, batch_img)

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


# import pytest
# @pytest.mark.parametrize(
#     'batch_size, num_epochs, learning_rate, param_group',
#     [(128, 3, 0.0001, True), ]
# )
# def test_resnet_train_ft(batch_size, num_epochs, learning_rate, param_group):
#     train_ft(batch_size, num_epochs, learning_rate, param_group)


if __name__ == '__main__':
    # train_ft(128, 5, 0.0001, True)
    # predict('sunflowers2.webp')
    batch_predict(batch_size=5, shows=15)

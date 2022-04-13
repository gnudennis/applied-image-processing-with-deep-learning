import json
import os

import torch
from PIL import Image
from torch import nn, optim
from torchvision import transforms, datasets

from aip import train_wrapper, predict_wrapper
from aip.models import lenet5
from aip.utils import get_dataloader_workers, try_all_gpus, show_images


def train(batch_size=128, num_epochs=5, learning_rate=1e-3):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get data root path
    dataset_root = os.path.join(root, 'dataset', 'cifar10')  # cifar10 set path
    assert os.path.exists(dataset_root), f'{dataset_root} path does not exist.'
    model_path = os.path.join(root, 'saved', 'lenet')
    assert os.path.exists(model_path), f'{model_path} path does not exist.'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    train_dataset = datasets.CIFAR10(dataset_root, train=True, download=True, transform=transform)
    valid_dataset = datasets.CIFAR10(dataset_root, train=False, download=True, transform=transform)
    print(f'using {len(train_dataset)} images for training, {len(valid_dataset)} images for validation.')

    # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(model_path, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)

    nw = get_dataloader_workers(batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    net = lenet5()
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)

    devices = try_all_gpus()
    saved_path = os.path.join(model_path, 'lenet5.pth')
    train_wrapper(net, train_loader, valid_loader, loss, optimizer, num_epochs, devices, saved_path)

    from matplotlib import pyplot as plt
    plt.show()


def predict(image_name):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get data root path
    dataset_root = os.path.join(root, 'dataset', 'cifar10')  # cifar10 set path
    assert os.path.exists(dataset_root), f'{dataset_root} path does not exist.'
    model_path = os.path.join(root, 'saved', 'lenet')
    assert os.path.exists(model_path), f'{model_path} path does not exist.'

    transform = transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    net = lenet5()
    model_weight_path = os.path.join(model_path, 'lenet5.pth')
    assert os.path.exists(model_weight_path), f'file {model_path} does not exist.'
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    # print('[missing_keys]:', *missing_keys, sep='\n')
    # print('[unexpected_keys]:', *unexpected_keys, sep='\n')

    # read cla_dict
    json_path = os.path.join(model_path, 'class_indices.json')
    assert os.path.exists(json_path), f'file: {json_path} dose not exist.'
    cla_dict = json.load(open(json_path, 'r'))

    test_image_path = os.path.join(dataset_root, 'pred', image_name)
    assert os.path.exists(test_image_path), f'file: {test_image_path} dose not exist.'

    pimg = Image.open(test_image_path)
    # plt.imshow(pimg)
    # [N, C, H, W]
    img = torch.unsqueeze(transform(pimg), dim=0)
    probs, classes = predict_wrapper(net, img)

    titles = [f'{os.path.splitext(os.path.basename(test_image_path))[0]}\n'
              f'({cla_dict[str(int(classes))]}:{float(probs):.3f})']
    show_images([pimg], 1, 1, titles, scale=2.5)


def batch_predict(batch_size=8, formats=['.jpg', '.jpeg', '.webp'], shows=12):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get data root path
    dataset_root = os.path.join(root, 'dataset', 'cifar10')  # cifar10 set path
    assert os.path.exists(dataset_root), f'{dataset_root} path does not exist.'
    model_path = os.path.join(root, 'saved', 'lenet')
    assert os.path.exists(model_path), f'{model_path} path does not exist.'

    transform = transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    net = lenet5()
    model_weight_path = os.path.join(model_path, 'lenet5.pth')
    assert os.path.exists(model_weight_path), f'file {model_path} does not exist.'
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    # print('[missing_keys]:', *missing_keys, sep='\n')
    # print('[unexpected_keys]:', *unexpected_keys, sep='\n')

    # read cla_dict
    json_path = os.path.join(model_path, 'class_indices.json')
    assert os.path.exists(json_path), f'file: {json_path} dose not exist.'
    cla_dict = json.load(open(json_path, 'r'))

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
    # train(128, 20, 1e-3)
    # predict('airplane1.jpeg')
    batch_predict(batch_size=5, shows=15)

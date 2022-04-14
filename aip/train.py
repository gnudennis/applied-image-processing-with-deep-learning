import sys

import torch
from torch import nn
from tqdm import tqdm

from .evaluate import accuracy, evaluate_accuracy_gpu
from .utils import try_all_gpus, Accumulator, Animator, Timer

__all__ = ['train_wrapper']


def train_batch(net, X, y, loss, optimizer, devices):
    """Train for a minibatch with mutiple GPUs"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    optimizer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    optimizer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_wrapper(net, train_loader, valid_loader, loss, optimizer, num_epochs, devices=try_all_gpus(),
                  saved_path=None):
    timer, num_batches = Timer(), len(train_loader)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    best_acc = 0.0
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = Accumulator(4)
        train_bar = tqdm(train_loader, file=sys.stdout, initial=1)

        for i, (features, labels) in enumerate(train_bar):
            step = train_bar.n
            timer.start()
            l, acc = train_batch(net, features, labels, loss, optimizer, devices)
            timer.stop()

            metric.add(l, acc, labels.shape[0], labels.numel())
            train_bar.desc = f'loss {metric[0] / metric[2]:.3f}, train acc ' f'{metric[1] / metric[3]:.3f}'

            if step % (num_batches // 5) == 0 or step == num_batches:
                animator.add(epoch + step / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))
        valid_acc = evaluate_accuracy_gpu(net, valid_loader)
        animator.add(epoch + 1, (None, None, valid_acc))

        if saved_path is not None and valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(net.module.state_dict(), saved_path)
            print(
                f'[best saved](epoch: {epoch + 1:<3})loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, valid acc {valid_acc:.3f}')

    print(f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, valid acc {valid_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')

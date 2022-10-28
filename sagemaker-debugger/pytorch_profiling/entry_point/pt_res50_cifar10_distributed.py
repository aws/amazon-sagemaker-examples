# Standard Library
import argparse
import time

# Third Party
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# First Party
from smdebug import modes
from smdebug.pytorch import get_hook


def train(batch_size, epoch, net, hook, device, local_rank):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=f"./data_{local_rank}", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    validset = torchvision.datasets.CIFAR10(
        root=f"./data_{local_rank}", train=False, download=True, transform=transform_valid
    )
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    loss_optim = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.9)

    epoch_times = []

    if hook:
        hook.register_loss(loss_optim)
    # train the model

    for i in range(epoch):
        print("START TRAINING")
        if hook:
            hook.set_mode(modes.TRAIN)
        start = time.time()
        net.train()
        train_loss = 0
        for _, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_optim(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print("START VALIDATING")
        if hook:
            hook.set_mode(modes.EVAL)
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(validloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = loss_optim(outputs, targets)
                val_loss += loss.item()

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print(
            "Epoch %d: train loss %.3f, val loss %.3f, in %.1f sec"
            % (i, train_loss, val_loss, epoch_time)
        )

    # calculate training time after all epoch
    p50 = np.percentile(epoch_times, 50)
    return p50


def main():
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--local_rank", type=int)
    opt = parser.parse_args()

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # Init hook
    hook = get_hook()

    # create model
    net = models.__dict__["resnext101_32x8d"](pretrained=False)
    device = torch.device("cuda")
    net.to(device)

    # Start the training.
    median_time = train(opt.batch_size, opt.epoch, net, hook, device, opt.local_rank)
    print("Median training time per Epoch=%.1f sec" % median_time)


if __name__ == "__main__":
    main()

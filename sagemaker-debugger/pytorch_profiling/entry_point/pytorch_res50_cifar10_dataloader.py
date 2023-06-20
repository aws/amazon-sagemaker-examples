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
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

transform_delay_in_seconds = 0.0
training_delay = 0.0


def transform_delay(img):
    time.sleep(transform_delay_in_seconds)
    return img


def train(args, net, device):
    hook = get_hook(create_if_not_exists=True)
    batch_size = args.batch_size
    epoch = args.epoch
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(transform_delay),
        ]
    )

    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(transform_delay),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
    )

    validset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_valid
    )
    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
    )

    loss_optim = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.9)

    epoch_times = []

    if hook:
        hook.register_module(net)
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
    global transform_delay_in_seconds
    global training_delay
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--transform_delay", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--training_delay", type=float, default=0.0)

    opt = parser.parse_args()

    opt.pin_memory = True if opt.pin_memory else False
    transform_delay_in_seconds = opt.transform_delay
    training_delay = opt.training_delay

    for key, value in vars(opt).items():
        print(f"{key}:{value}")
    # create model
    net = models.__dict__[opt.model](pretrained=True)
    if opt.gpu == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net.to(device)

    # Start the training.
    median_time = train(opt, net, device)
    print("Median training time per Epoch=%.1f sec" % median_time)


if __name__ == "__main__":
    main()

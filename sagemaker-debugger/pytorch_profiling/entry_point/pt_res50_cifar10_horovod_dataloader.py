# Standard Library
import argparse
import time

# Third Party
import horovod.torch as hvd
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


def train(batch_size, epoch, net, hook, args, local_rank):
    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
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
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=hvd.size(), rank=hvd.rank()
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, **kwargs
    )

    validset = torchvision.datasets.CIFAR10(
        root=f"./data_{local_rank}", train=False, download=True, transform=transform_valid
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        validset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, sampler=test_sampler, **kwargs
    )

    loss_optim = nn.CrossEntropyLoss()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(net.parameters(), lr=0.01 * hvd.size(), momentum=0.5)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=net.named_parameters(), compression=compression
    )

    epoch_times = []

    if hook:
        hook.register_loss(loss_optim)
    # train the model

    for i in range(epoch):
        print("START TRAINING")
        if hook:
            hook.set_mode(modes.TRAIN)
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(i)
        start = time.time()
        net.train()
        train_loss = 0
        for _, (inputs, targets) in enumerate(trainloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_optim(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print("START VALIDATING")
        if hook:
            hook.register_module(net)
            hook.set_mode(modes.EVAL)
        test_sampler.set_epoch(i)
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(validloader):
                if args.cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
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
    parser.add_argument(
        "--epochs", type=int, default=5, metavar="N", help="number of epochs to train (default: 5)"
    )
    parser.add_argument("--use_only_cpu", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="resnet50")
    args = parser.parse_args()

    args.cuda = not args.use_only_cpu and torch.cuda.is_available()

    batch_size = args.batch_size
    seed = 42

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(seed)

    local_rank = hvd.local_rank()
    # create model
    net = models.__dict__[args.model](pretrained=False)
    if args.cuda:
        net.cuda()
    # Init hook
    hook = get_hook()

    # Start the training.
    median_time = train(batch_size, args.epochs, net, hook, args, local_rank)
    print("Median training time per Epoch=%.1f sec" % median_time)


if __name__ == "__main__":
    main()

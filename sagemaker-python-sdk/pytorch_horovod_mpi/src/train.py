"""Functions for training a simple neural network."""
import argparse
import os

import torch
import torch.multiprocessing as mp
from torchvision import transforms, datasets
import horovod.torch as hvd

from model import Net


def main(args):
    """Main program."""
    if not torch.cuda.is_available():
        raise RuntimeError("Expected CUDA-capable device but found none.")

    init(args.seed)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=args.data_dir,
                             train=True,
                             transform=transform)

    model = Net().to("cuda")

    train(model,
          dataset,
          batch_size=args.batch_size,
          num_epochs=args.epochs,
          momentum=args.momentum,
          learning_rate=args.lr)

    save(model, args.model_dir)


def init(seed):
    """Initialize Horovod and PyTorch for distributed training."""
    hvd.init()
    torch.manual_seed(seed)
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)


def train(model,
          dataset,
          batch_size=4,
          num_epochs=2,
          momentum=0.9,
          learning_rate=0.001):
    """Trains a model over a dataset.

    Arguments:
        model (nn.Module): A PyTorch model.
        dataset (torch.utils.data.Dataset): The training dataset.
        batch_size (int): The number of samples per batch to predict.
        num_epochs (int): The number of times the dataset will be iterated over.
        momentum (float): The SGD momentum factor.
        learning_rate (float): The learning rate.
    """
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank())

    if forkserver_is_available():
        multiprocessing_context = "forkserver"
    else:
        multiprocessing_context = "fork"

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        multiprocessing_context=multiprocessing_context)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate * hvd.local_size(),
                                momentum=momentum)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters())

    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for inputs, labels in dataloader:
            optimizer.zero_grad()

            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()


def save(model, directory):
    """Saves the parameters of a trained model to a file.

    Arguments:
        model (nn.Module): A PyTorch model.
        directory (str): Path to the directory where the model will be saved.
    """
    if hvd.rank() != 0:
        return
    state_dict = model.cpu().state_dict()
    file_path = os.path.join(directory, "model.pth")
    torch.save(state_dict, file_path)


def forkserver_is_available():
    """Returns true if forkserver can be used as a multiprocessing context."""
    return hasattr(
        mp, '_supports_context'
    ) and mp._supports_context and 'forkserver' in mp.get_all_start_methods()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument('--model-dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir',
                        type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])

    args = parser.parse_args()
    main(args)

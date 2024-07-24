import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

from sagemaker.remote_function import remote
from model import Net
from load_data import load_data


# Set path to config file
os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

def train(model, device, train_loader, optimizer, epoch, log_interval, dry_run):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


@remote(include_local_workdir=True)
def perform_train(train_data,
                  test_data,
                  *,
                  batch_size: int = 64,
                  test_batch_size: int = 1000,
                  epochs: int = 3,
                  lr: float = 1.0,
                  gamma: float = 0.7,
                  no_cuda: bool = True,
                  no_mps: bool = True,
                  dry_run: bool = False,
                  seed: int = 1,
                  log_interval: int = 10,
                  ):
    """PyTorch MNIST Example

    :param train_data: the training data set
    :param test_data: the test data set
    :param batch_size: input batch size for training (default: 64)
    :param test_batch_size: input batch size for testing (default: 1000)
    :param epochs: number of epochs to train (default: 14)
    :param lr: learning rate (default: 1.0)
    :param gamma: Learning rate step gamma (default: 0.7)
    :param no_cuda: disables CUDA training
    :param no_mps: disables macOS GPU training
    :param dry_run: quickly check a single pass
    :param seed: random seed (default: 1)
    :param log_interval: how many batches to wait before logging training status
    :return: the trained model
    """

    use_cuda = not no_cuda and torch.cuda.is_available()
    use_mps = not no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval, dry_run)
        test(model, device, test_loader)
        scheduler.step()

    return model


if __name__ == "__main__":
    training_data, test_data = load_data()

    model = perform_train(training_data, test_data, dry_run=True)

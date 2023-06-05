import boto3
from torchvision import datasets, transforms


def load_data():
    datasets.MNIST.mirrors = [
        f"https://sagemaker-example-files-prod-{boto3.session.Session().region_name}.s3.amazonaws.com/datasets/image/MNIST/"
    ]

    train_set = datasets.MNIST(
        "./data",
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        download=True,
    )

    test_set = datasets.MNIST(
        "./data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        download=True,
    )

    return train_set, test_set

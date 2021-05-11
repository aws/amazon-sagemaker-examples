import argparse
import copy
import os
import pathlib
import shutil
import time

import torch
import torchvision as tv

# the training fuction is based off the sample training fuction provided
# by Pytorch in their transfer learning tutorial:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


def train(model, criterion, optimizer, scheduler, epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)

    args, _ = parser.parse_known_args()

    data_dir = pathlib.Path("/opt/ml/input/data")

    # define transformations
    data_transforms = {
        "train": tv.transforms.Compose(
            [
                tv.transforms.RandomResizedCrop(224),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomVerticalFlip(p=0.5),
                tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                tv.transforms.ToTensor(),
            ]
        ),
        "val": tv.transforms.Compose([tv.transforms.CenterCrop(224), tv.transforms.ToTensor()]),
    }

    # create datasets and dataloaders
    splits = ["train", "val"]
    datasets = {}
    for s in splits:
        datasets[s] = tv.datasets.ImageFolder(root=data_dir / s, transform=data_transforms[s])

    dataloaders = {}
    for s in splits:
        dataloaders[s] = torch.utils.data.DataLoader(
            datasets[s], batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )

    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}
    num_classes = len(datasets["train"].classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training with: {device}")

    model = tv.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train(model, criterion, optimizer, exp_lr_scheduler, epochs=args.epochs)

    torch.save(model.state_dict(), "/opt/ml/model/model.pt")

from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.functional as F
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import backbones.ResNet
from load_data import build
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from confusion_matrix import Confusion_Matrix
from util import tsne_feature_visualization, D2_images_sar_plot


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True


def finetune(args, model, train_loader, test_loader, criterion, optimizer, device):
    model.train()
    print("start_training")
    for epoch in range(1, args.epochs + 1):
        Loss, train_correct = 0, 0
        init_time = time.time()
        for idx, (data, target) in enumerate(train_loader):
            time_pass = time.time() - init_time
            if idx == 0:
                print(time_pass)
            img, label = data.to(device), target.to(device)
            feature, output = model(img)
            predictions = torch.max(output, dim=1)[1]
            train_correct += predictions.eq(label.data.view_as(predictions)).sum()
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Loss += loss.item() * data.size(0)

        Loss /= len(train_loader.dataset)
        acc = 100.0 * train_correct / len(train_loader.dataset)
        writer.add_scalar("Train/Loss", Loss, epoch)
        writer.add_scalar("Train/Acc", acc, epoch)
        print(
            "Train Epoch: [{}/{}]\t Loss: {:.6f}\t Accuracy: {}/{} ({:.2f}%)".format(
                epoch, args.epochs, loss, train_correct, len(train_loader.dataset), acc
            )
        )
        (
            test_true,
            test_pred,
            test_correct,
            len_test_dataset,
            test_loss,
            test_acc,
        ) = test(model, test_loader, criterion, device)
        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/Acc", test_acc, epoch)
        print(
            "Test Accuracy: {}/{} ({:.2f}%)".format(
                test_correct, len_test_dataset, test_acc
            )
        )
    true_label = torch.hstack(test_true)
    pred_label = torch.hstack(test_pred)
    cm = confusion_matrix(true_label.data.cpu().numpy(), pred_label.data.cpu().numpy())
    classes = test_loader.dataset.classes
    Confusion_Matrix(num_classes=args.num_classes, label=classes, matrix=cm)


@torch.no_grad()
def test(model, test_loader, criterion, device):
    model.eval()
    test_correct, Test_Loss = 0, 0
    test_pred, test_true = [], []
    for data, target in test_loader:
        img, label = data.to(device), target.to(device)
        feature, output = model(img)
        test_loss = criterion(output, label)
        predictions = torch.max(output, dim=1)[1]
        test_pred.append(predictions)
        test_true.append(label)
        test_correct += predictions.eq(label.data.view_as(predictions)).sum()
    test_acc = 100.0 * test_correct / len(test_loader.dataset)
    Test_Loss += test_loss.item() * data.size(0)

    return (
        test_true,
        test_pred,
        test_correct,
        len(test_loader.dataset),
        Test_Loss,
        test_acc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a retrieval network")
    parser.add_argument("--model_name", help="model", default="resnet50", type=str)
    parser.add_argument("--cuda_id", help="cuda id", default="1", type=str)
    parser.add_argument(
        "--data_root",
        help="dataset root path",
        default="/home/xjw/jianwen/data/",
        type=str,
    )
    parser.add_argument("--dataset", help="dataset", default="chips", type=str)
    parser.add_argument("--lr", help="learning rate", default=1e-4, type=float)
    parser.add_argument("--decay", help="weight decay", default=5e-4, type=float)
    parser.add_argument("--epochs", help="num_epochs", default=200, type=int)
    parser.add_argument("--batch_size", help="batch_size", default=100, type=int)
    parser.add_argument("--workers", help="workers of dataloader", default=2, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter()

    train_loader = build.build_data(args, is_train=True)
    test_loader = build.build_data(args, is_train=False)

    model = backbones.ResNet.resnet50(
        pretrained=True, num_classes=len(train_loader.dataset.classes)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    since = time.time()
    finetune(args, model, train_loader, test_loader, criterion, optimizer, device)
    writer.close()
    time_pass = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(time_pass // 60, time_pass % 60)
    )

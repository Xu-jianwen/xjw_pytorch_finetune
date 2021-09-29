from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import os
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import backbones
from load_data import build
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from confusion_matrix import Confusion_Matrix
from util import tsne_feature_visualization, D2_images_sar_plot, set_bn_eval


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True


def finetune(args, model, train_loader, test_loader, criterion, optimizer, device):
    model.train()
    # model.apply(set_bn_eval)
    print("start_training")
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        Loss, train_correct = 0, 0
        for idx, (data, target) in enumerate(train_loader):
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
        if test_acc >= best_acc:
            best_acc = test_acc
            best_model = model
            print("Best acc :{:.2f}%, at Epoch {}".format(best_acc, epoch))
        else:
            print("Best acc is :{:.2f}%".format(best_acc))
    torch.save(best_model, args.dataset + "_best_model.pth")
    true_label = torch.hstack(test_true)
    pred_label = torch.hstack(test_pred)
    cm = confusion_matrix(true_label.data.cpu().numpy(), pred_label.data.cpu().numpy())
    classes = test_loader.dataset.classes
    Confusion_Matrix(num_classes=len(classes), label=classes, matrix=cm)


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
    parser = argparse.ArgumentParser(description="Train a CNN")
    parser.add_argument("--model_name", help="model", default="resnet50", type=str)
    parser.add_argument("--embedding_size", help="model", default="256", type=int)
    parser.add_argument("--embedding", help="model", default=True, type=bool)
    parser.add_argument("--cuda_id", help="cuda id", default="1", type=str)
    parser.add_argument(
        "--data_root",
        help="dataset root path",
        default="/home/xjw/jianwen/data/",
        type=str,
    )
    parser.add_argument("--dataset", help="dataset", default="FGSC23", type=str)
    parser.add_argument("--lr", help="learning rate", default=1e-3, type=float)
    parser.add_argument("--decay", help="weight decay", default=5e-4, type=float)
    parser.add_argument("--momentum", help="SGD momentum", default=0.9, type=float)
    parser.add_argument("--epochs", help="num_epochs", default=400, type=int)
    parser.add_argument("--batch_size", help="batch_size", default=100, type=int)
    parser.add_argument("--workers", help="workers of dataloader", default=4, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter()

    train_loader = build.build_data(args, is_train=True)
    test_loader = build.build_data(args, is_train=False)

    # model = backbones.ResNet.resnet50(
    #     pretrained=True, num_classes=len(train_loader.dataset.classes)
    # )
    model = backbones.create(name=args.model_name, pretrained=True, model_name=args.model_name,
                             dim=args.embedding_size, num_class=len(train_loader.dataset.classes), embedding=args.embedding)
    model.to(device)
    # print(model)
    # model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.decay, momentum=args.momentum)

    since = time.time()
    finetune(args, model, train_loader, test_loader, criterion, optimizer, device)
    writer.close()
    time_pass = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(time_pass // 60, time_pass % 60)
    )

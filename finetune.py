from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import backbones
from load_data import build
from torch.utils.data import DataLoader
import copy
import pandas as pd
from sklearn.metrics import confusion_matrix
from util import cm_plot, tsne_feature_visualization, tsne_plot, set_bn_eval


torch.backends.cudnn.benchmark = True


def finetune(args, model, train_loader, test_loader, criterion, optimizer, device):
    model.train()
    # model.apply(set_bn_eval)
    print("start_training")
    best_acc = 0
    test_accs = []
    for epoch in range(1, args.epochs + 1):
        Loss, train_correct = 0, 0
        print("Learning Rate:{:.6f}".format(optimizer.param_groups[0]['lr']))
        for idx, (data, target) in enumerate(train_loader):
            img, label = data.to(device), target.to(device)
            # feature, output = model(img)
            ft, logits = model(img)
            if args.num_centers is not None:
                output = proxies_reducer(args.num_centers, logits)
            else:
                output = logits
            predictions = torch.max(output, dim=1)[1]
            train_correct += predictions.eq(label.data.view_as(predictions)).sum()
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

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
        ) = test(args, model, test_loader, criterion, device)
        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/Acc", test_acc, epoch)
        test_accs.append(test_acc.item())
        print(
            "Test Accuracy: {}/{} ({:.2f}%)".format(
                test_correct, len_test_dataset, test_acc
            )
        )
        if test_acc >= best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
            print("Best acc :{:.2f}%, at Epoch {}".format(best_acc, epoch))
        else:
            print("Best acc is :{:.2f}%".format(best_acc))
    accs = pd.DataFrame(columns=["acc"], data=test_accs)
    os.makedirs("acc_csv/", exist_ok=True)
    accs.to_csv("acc_csv/" + args.model_name + "_" + args.dataset + "_test_acc.csv")
    os.makedirs("ckps/" + args.dataset, exist_ok=True)
    torch.save(
        best_model,
        os.path.join("ckps/" + args.dataset, args.model_name + "_best_model.pth"),
    )
    torch.save(
        model,
        os.path.join("ckps/" + args.dataset, args.model_name + ".pth"),
    )
    true_label = torch.hstack(test_true)
    pred_label = torch.hstack(test_pred)
    cm = confusion_matrix(true_label.data.cpu().numpy(), pred_label.data.cpu().numpy())
    classes = test_loader.dataset.classes
    cm_plot(num_classes=len(classes), label=classes, matrix=cm)


@torch.no_grad()
def test(args, model, test_loader, criterion, device):
    model.eval()
    test_correct, Test_Loss = 0, 0
    test_pred, test_true = [], []
    for data, target in test_loader:
        img, label = data.to(device), target.to(device)
        # feature, output = model(img)
        ft, logits = model(img)
        if args.num_centers is not None:
            output = proxies_reducer(args.num_centers, logits)
        else:
            output = logits
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


def proxies_reducer(num_centers, logit):
    logit = logit.view(-1, len(train_loader.dataset.classes), num_centers)
    prob = F.softmax(logit * 0.1, dim=2)
    sim_to_classes = torch.sum(prob * logit, dim=2)
    return sim_to_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN")
    parser.add_argument("--model_name", help="model", default="AlexNet", type=str)
    parser.add_argument("--embedding_size", default="512", type=int)
    parser.add_argument("--embedding", default=True, type=bool)
    parser.add_argument("--num_centers", default=None)
    parser.add_argument("--cuda_id", help="cuda id", default="1", type=str)
    parser.add_argument(
        "--data_root",
        default="/home/xjw/jianwen/data/ship_align/",
        type=str,
    )
    parser.add_argument("--dataset", help="dataset", default="original_img", type=str)
    parser.add_argument("--lr", help="learning rate", default=1e-4, type=float)
    parser.add_argument("--decay", help="weight decay", default=5e-4, type=float)
    parser.add_argument("--momentum", help="SGD momentum", default=0.9, type=float)
    parser.add_argument("--epochs", help="num_epochs", default=100, type=int)
    parser.add_argument("--batch_size", help="batch_size", default=100, type=int)
    parser.add_argument("--workers", help="workers of dataloader", default=4, type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter()

    train_loader = build.build_data(
        args,
        path_list=os.path.join(args.data_root, args.dataset, "train.txt"),
        is_train=True,
    )
    test_loader = build.build_data(
        args,
        path_list=os.path.join(args.data_root, args.dataset, "test.txt"),
        is_train=False,
    )

    if args.num_centers is None:
        num_centers = 1
    else:
        num_centers = args.num_centers
    model = backbones.create(
        name=args.model_name,
        pretrained=True,
        # dim=args.embedding_size,
        num_classes=len(train_loader.dataset.classes) * num_centers,
        # embedding=args.embedding,
    )
    model.to(device)
    # model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50 * len(train_loader)], gamma=0.1
    )

    since = time.time()
    finetune(args, model, train_loader, test_loader, criterion, optimizer, device)
    writer.close()
    time_pass = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(time_pass // 60, time_pass % 60)
    )

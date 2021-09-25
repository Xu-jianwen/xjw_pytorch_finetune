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
import data_loader
import argparse
import backbones.ResNet
from dataset import ship_dataset
from torch.utils.data import DataLoader
from confusion_matrix import Confusion_Matrix
from util import tsne_feature_visualization, D2_images_sar_plot


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True
def finetune(args, model, train_loader, test_loader, criterion, optimizer, device):
    model.train()
    for epoch in range(1, args.epochs+1):
        Loss, train_correct = 0, 0
        for data, target in train_loader:
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
        acc = 100. * train_correct / len(train_loader.dataset)
        print('Train Epoch: [{}/{}]\t Loss: {:.6f}\t Accuracy: {}/{} ({:.2f}%)'.format(epoch, args.epochs, loss, train_correct, len(train_loader.dataset), acc))
        test_correct, len_test_dataset, test_acc = test(args, model, test_loader, device)
        print('Test Accuracy: {}/{} ({:.2f}%)'.format(test_correct, len_test_dataset, test_acc))
    # return Loss, train_correct, len(train_loader.dataset), total_train_accuracy


@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    test_correct = 0
    for data, target in test_loader:
        img, label = data.to(device), target.to(device)
        feature, output = model(img)
        predictions = torch.max(output, dim=1)[1]
        test_correct += predictions.eq(label.data.view_as(predictions)).sum()
    test_acc = 100. * test_correct/len(test_loader.dataset)
    return test_correct, len(test_loader.dataset), test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a retrieval network')
    parser.add_argument('--model_name', help='model', default="resnet50", type=str)
    # parser.add_argument('--device', help='cuda or cpu', default="cuda", type=str)
    parser.add_argument('--data_root', help='dataset root path', default="/home/pc001/jianwen/data/", type=str)
    parser.add_argument('--dataset', help='dataset', default=None, type=str)
    parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--decay', help='weight decay', default=5e-4, type=float)
    parser.add_argument('--epochs', help='num_epochs', default=200, type=int)
    parser.add_argument('--num_classes', help='num_classes', default=28, type=int)
    parser.add_argument('--batch_size', help='batch_size', default=100, type=int)
    parser.add_argument('--workers', help='workers of dataloader', default=8, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = backbones.ResNet.resnet50(pretrained=True, num_classes=args.num_classes)
    model.to(device)

    data = ship_dataset(root=None)
    train_loader = DataLoader(data.train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(data.test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    # train_loader = data_loader.load_data_train(args.data_root, "ship28", args.batch_size, kwargs=None)
    # test_loader = data_loader.load_data_train(args.data_root, "ship28", args.batch_size, kwargs=None)
    criterion = nn.CrossEntropyLoss()
    optimizer =  optim.Adam(model.parameters(), lr=args.lr , weight_decay=args.decay)
    since = time.time()
    finetune(args, model, train_loader, test_loader, criterion, optimizer, device)
    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
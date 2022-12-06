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
from sklearn.metrics import confusion_matrix
from util import cm_plot, set_bn_eval, proxies_reducer, distance_matrix
from losses import LMMDLoss
from pytorch_metric_learning.losses import MultiSimilarityLoss, CircleLoss, ContrastiveLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner, DistanceWeightedMiner


torch.backends.cudnn.benchmark = True


def finetune(args, model, train_loader, test_loader, criterion, miner, optimizer, device):
    model.train()
    # model.apply(set_bn_eval)
    print(args.dataset + " start training")
    best_acc = 0
    test_accs = []
    for epoch in range(1, args.epoches + 1):
        Loss, train_correct = 0, 0
        SLMMD = 0
        print("Learning Rate:{:.6f}".format(optimizer.param_groups[0]["lr"]))
        for idx, (data, target) in enumerate(train_loader):
            img, label = data.to(device), target.to(device)
            # feature, output = model(img)
            ft, logits = model(img)
            # normed_weights = F.normalize(model.classifier.weight, p=2, dim=1)
            normed_weights = F.normalize(model.module.classifier.weight, p=2, dim=1)

            logits = ft @ normed_weights.t()
            
            proxies_mask = torch.tensor(
                int(logits.size(1) / len(train_loader.dataset.classes))
                * list(range(len(train_loader.dataset.classes))),
                device=label.device,
            )
            predictions = (torch.max(logits, dim=1)[1]) % len(
                train_loader.dataset.classes
            )
            train_correct += predictions.eq(label.data.view_as(predictions)).sum()

            embeddings = torch.cat((ft, normed_weights), dim=0)
            labels = torch.cat((label, proxies_mask))
            indices_tuple = miner(embeddings, labels)
            cls_loss = criterion(embeddings, labels, indices_tuple)
            mmd = mmd_loss(ft, normed_weights, label, proxies_mask)
            loss = cls_loss + args.alpha * 0.001 * mmd
            # indices_tuple = miner(ft, label)
            # pair_loss = criterion(ft, label, indices_tuple)
            # cls_loss = ce(logits, label)
            # loss = cls_loss + pair_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            Loss += cls_loss.item() * data.size(0)
            SLMMD += mmd.item()

        Loss /= len(train_loader.dataset)
        SLMMD /= len(train_loader)
        # acc = 100.0 * train_correct / len(train_loader.dataset)
        acc = 100.0 * train_correct / (len(train_loader) * args.batch_size)
        writer.add_scalar("Train/Loss", Loss, epoch)
        writer.add_scalar("Train/MMD", SLMMD, epoch)
        writer.add_scalar("Train/Acc", acc, epoch)
        print(
            "Train Epoch: [{}/{}]\t Loss: {:.6f}\t Accuracy: {}/{} ({:.2f}%)".format(
                epoch, args.epoches, Loss, train_correct, len(train_loader.dataset), acc
            )
        )
        (
            test_true,
            test_pred,
            test_correct,
            len_test_dataset,
            test_loss,
            test_acc,
        ) = validate(args, model, test_loader, criterion, device)
        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/Acc", test_acc, epoch)
        test_accs.append(test_acc.item())
        print(
            "Test Accuracy: {}/{} ({:.2f}%) loss:{:.4f}".format(
                test_correct, len_test_dataset, test_acc, SLMMD
            )
        )
        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = copy.deepcopy(epoch)
            best_model = copy.deepcopy(model)
            print("Best acc :{:.2f}%, at Epoch {}".format(best_acc, best_epoch))
        else:
            print("Best acc :{:.2f}%, at Epoch {}".format(best_acc, best_epoch))

    os.makedirs("ckps/" + args.dataset, exist_ok=True)
    torch.save(
        best_model,
        os.path.join(
            "ckps/" + args.dataset, args.model_name +"_alpha_"+str(args.alpha)+ "_cen_2_best_model.pth"
        ),
    )
    # torch.save(
    #     best_model,
    #     os.path.join(
    #         "ckps/" + args.dataset, args.model_name + "_circle_best_model.pth"
    #     ),
    # )
    # torch.save(
    #     model,
    #     os.path.join("ckps/" + args.dataset, args.model_name + "_ce.pth"),
    # )
    true_label = torch.hstack(test_true)
    pred_label = torch.hstack(test_pred)
    cm = confusion_matrix(true_label.data.cpu().numpy(), pred_label.data.cpu().numpy())
    classes = test_loader.dataset.classes
    cm_plot(num_classes=len(classes), label=classes, matrix=cm)


@torch.no_grad()
def validate(args, model, test_loader, criterion, device):
    model.eval()
    test_correct, Test_Loss = 0, 0
    test_pred, test_true = [], []
    for data, target in test_loader:
        img, label = data.to(device), target.to(device)
        # feature, output = model(img)
        ft, logits = model(img)
        # Normed_Weights = F.normalize(model.classifier.weight, p=2, dim=1)
        Normed_Weights = F.normalize(model.module.classifier.weight, p=2, dim=1)

        logits = ft @ Normed_Weights.t()
        output = proxies_reducer(
            len(test_loader.dataset.classes), args.num_centers, logits
        )
        # output = logits
        # test_loss = criterion(output, label)
        # test_loss = criterion(Normed_Weights, ft, label)
        # test_loss = criterion(logits, label)
        predictions = (torch.max(logits, dim=1)[1]) % len(test_loader.dataset.classes)
        test_pred.append(predictions)
        test_true.append(label)
        test_correct += predictions.eq(label.data.view_as(predictions)).sum()
        # Test_Loss += test_loss.item() * data.size(0)
    test_acc = 100.0 * test_correct / len(test_loader.dataset)
    Test_Loss /= len(test_loader.dataset)

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
    parser.add_argument("--model_name", help="model", default="resnet34", type=str)
    parser.add_argument("--embedding_size", default=128, type=int)
    parser.add_argument("--embedding", default=True, type=bool)
    parser.add_argument("--num_centers", default=4, type=int)
    parser.add_argument("--alpha", default=50, type=int)
    parser.add_argument("--cuda_id", help="cuda id", default="1", type=str)
    parser.add_argument(
        "--data_root",
        default="/home/xjw/jianwen/data/",
        type=str,
    )
    parser.add_argument("--dataset", help="dataset", default="ShipRSImageNet_pad", type=str)
    parser.add_argument("--lr", help="learning rate", default=3e-5, type=float)
    parser.add_argument("--decay", help="weight decay", default=5e-4, type=float)
    parser.add_argument("--momentum", help="SGD momentum", default=0.9, type=float)
    parser.add_argument("--epoches", help="num_epoches", default=200, type=int)
    parser.add_argument("--batch_size", help="batch_size", default=128, type=int)
    parser.add_argument("--workers", help="workers of dataloader", default=4, type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter()

    train_loader = build.build_data(
        args,
        path_list=os.path.join(args.data_root, args.dataset, "train.txt"),
        is_train=True,
        pk_sample=True, 
    )
    test_loader = build.build_data(
        args,
        path_list=os.path.join(args.data_root, args.dataset, "val.txt"),
        is_train=False,
    )

    if args.num_centers is None:
        num_centers = 1
    else:
        num_centers = args.num_centers
    model = backbones.create(
        name=args.model_name,
        pretrained=True,
        dim=args.embedding_size,
        num_classes=len(train_loader.dataset.classes) * num_centers,
        embedding=args.embedding,
    )
    # model.to(device)
    model = model.cuda()
    model = nn.DataParallel(model)
    ce = nn.CrossEntropyLoss()
    mmd_loss = LMMDLoss(num_class=len(train_loader.dataset.classes))
    criterion = MultiSimilarityLoss()
    miner = MultiSimilarityMiner(epsilon=0.1)
    # miner = None
    # miner = DistanceWeightedMiner(cutoff=0.5, nonzero_loss_cutoff=1.4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10 * len(train_loader)], gamma=0.15
    )

    since = time.time()
    finetune(args, model, train_loader, test_loader, criterion, miner, optimizer, device)
    writer.close()
    time_pass = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(time_pass // 60, time_pass % 60)
    )

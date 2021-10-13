from __future__ import print_function, absolute_import
import torch
import os
import torch.utils
import argparse
from load_data import build
from sklearn.metrics import confusion_matrix
from util import cm_plot


parser = argparse.ArgumentParser(description="finetune a CNN")
parser.add_argument("--cuda_id", help="cuda id", default="1", type=str)
parser.add_argument(
    "--data_root",
    help="dataset root path",
    default="/home/xjw/jianwen/data/",
    type=str,
)
parser.add_argument("--dataset", help="dataset", default="ori_bgd", type=str)
parser.add_argument("--batch_size", help="batch_size", default=100, type=int)
parser.add_argument("--workers", help="workers of dataloader", default=2, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
device = "cuda" if torch.cuda.is_available() else "cpu"

test_loader = build.build_data(args, is_train=False)
ckp_path = os.path.join("ckps/" + args.dataset, args.dataset + "_best_model.pth")
model = torch.load(ckp_path)
model.to(device)


with torch.no_grad():
    test_correct = 0
    test_pred, test_true = [], []
    for data, target in test_loader:
        img, label = data.to(device), target.to(device)
        feature, output = model(img)
        predictions = torch.max(output, dim=1)[1]
        test_correct += predictions.eq(label.data.view_as(predictions)).sum()
        test_pred.append(predictions)
        test_true.append(label)
    test_acc = 100.0 * test_correct / len(test_loader.dataset)
    print(
        "Test Accuracy: {}/{} ({:.2f}%)".format(
            test_correct, len(test_loader.dataset), test_acc
        )
    )
    true_label = torch.hstack(test_true)
    pred_label = torch.hstack(test_pred)
    cm = confusion_matrix(true_label.data.cpu().numpy(), pred_label.data.cpu().numpy())
    classes = test_loader.dataset.classes

    os.makedirs("confusion_matrixs/", exist_ok=True)
    cm_plot(
        num_classes=len(classes),
        label=classes,
        matrix=cm,
        fig_name="confusion_matrixs/" + args.dataset + "_",
    )

import torch
import numpy as np
import os
import torch.utils
import argparse
from load_data import build
from sklearn.metrics import confusion_matrix
from util import cm_plot, tsne_feature_visualization, tsne_plot, proxies_reducer
from vis_tsne import VisTSNE


parser = argparse.ArgumentParser(description="finetune a CNN")
parser.add_argument("--cuda_id", help="cuda id", default="0", type=str)
parser.add_argument("--num_centers", default=None)
parser.add_argument("--model_name", help="model", default="resnet50", type=str)
parser.add_argument(
    "--data_root",
    help="dataset root path",
    default="/home/xjw/jianwen/data/",
    type=str,
)
parser.add_argument("--dataset", help="dataset", default="FGSC23", type=str)
parser.add_argument("--batch_size", help="batch_size", default=100, type=int)
parser.add_argument("--workers", help="workers of dataloader", default=2, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
device = "cuda" if torch.cuda.is_available() else "cpu"

test_loader = build.build_data(
    args,
    path_list=os.path.join(args.data_root, args.dataset, "test.txt"),
    is_train=False,
)

ckp_path = os.path.join("ckps/" + args.dataset, args.model_name + ".pth")
# ckp_path = os.path.join("ckps/" + args.dataset, args.model_name + "_mp_best_model.pth")

model = torch.load(ckp_path)
model.to(device)

with torch.no_grad():
    test_correct = 0
    test_pred, test_true = [], []
    fts = []
    for data, target in test_loader:
        img, label = data.to(device), target.to(device)
        feature, output = model(img)
        normed_weights = torch.nn.functional.normalize(
            model.classifier.weight, p=2, dim=1
        )
        if model.classifier.out_features > len(test_loader.dataset.classes):
            logits = feature @ normed_weights.t()
            output = proxies_reducer(
                len(test_loader.dataset.classes),
                int(model.classifier.out_features / len(test_loader.dataset.classes)),
                logits,
            )
        else:
            output = feature @ normed_weights.t()
        predictions = torch.max(output, dim=1)[1]
        test_correct += predictions.eq(label.data.view_as(predictions)).sum()
        test_pred.append(predictions)
        test_true.append(label)
        fts.append(feature)
    features = torch.vstack(fts).data.cpu().numpy()
    test_acc = 100.0 * test_correct / len(test_loader.dataset)
    print(
        "Test Accuracy: {}/{} ({:.2f}%)".format(
            test_correct, len(test_loader.dataset), test_acc
        )
    )
    true_label = torch.hstack(test_true)
    pred_label = torch.hstack(test_pred)

    mask = (true_label != pred_label).tolist()
    imgs = test_loader.dataset.path_list
    errors = np.array(imgs)[mask].tolist()
    print("mispredicted samples:\n{:}".format(errors))

    cm = confusion_matrix(true_label.data.cpu().numpy(), pred_label.data.cpu().numpy())
    classes = test_loader.dataset.classes

    os.makedirs("confusion_matrixs/", exist_ok=True)
    cm_plot(
        num_classes=len(classes),
        label=classes,
        matrix=cm,
        fig_name="confusion_matrixs/" + args.dataset + "_",
    )
    path_list = [
        os.path.join(test_loader.dataset.root, i) for i in test_loader.dataset.path_list
    ]
    tsne = VisTSNE(feat=features, path_list=path_list)
    features = tsne_feature_visualization(features, n_components=2)
    tsne_plot(
        name=args.dataset,
        features=features,
        labels=true_label.data.cpu().numpy(),
        classes=test_loader.dataset.classes,
    )

    tsne.vis_tsne(
        feats=features, img_list=path_list, grid=[18, 32], save_path="tsne.png"
    )

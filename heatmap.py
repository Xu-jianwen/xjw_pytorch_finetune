import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from util import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
img_dir = "/home/xjw/jianwen/data/chips_bgd/Bulk_Carrier"
img_name = [
    "Bulk_Carrier_001.bmp",
    "Bulk_Carrier_011.bmp",
    "Bulk_Carrier_100.bmp",
]
img_path = [os.path.join(img_dir, i) for i in img_name]

pil_imgs = [PIL.Image.open(img) for img in img_path]

# normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
torch_imgs = [
    (
        torch.from_numpy(np.asarray(pil_img))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255)
        .cuda()
    )
    for pil_img in pil_imgs
]
torch_imgs = [
    F.upsample(torch_img, size=(224, 224), mode="bilinear", align_corners=False)
    for torch_img in torch_imgs
]
normed_torch_imgs = torch_imgs  # normalizer(torch_img)

resnet = torch.load("chips_bgd_best_model.pth")
resnet.eval(), resnet.cuda()

resnet_model_dict = dict(
    type="resnet", arch=resnet, layer_name="layer4", input_size=(224, 224)
)
# resnet_gradcam = GradCAM(resnet_model_dict, True)
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)

images = []
for normed_torch_img in normed_torch_imgs:
    # mask, _ = gradcam(normed_torch_img)
    # heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = resnet_gradcampp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, normed_torch_img)

    # images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
    images.append(
        torch.stack([normed_torch_img.squeeze().cpu(), heatmap_pp, result_pp], 0)
    )

images = make_grid(torch.vstack(images), nrow=3)

output_dir = "output/" + img_dir.split("/")[-1]
os.makedirs(output_dir, exist_ok=True)
output_name = img_dir.split("/")[-2] + ".bmp"
output_path = os.path.join(output_dir, output_name)

save_image(images, output_path)

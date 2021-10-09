import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

from util import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
img_dir = "/home/xjw/jianwen/data/mbr/Bulk_Carrier"
# img_name = 'collies.JPG'
# img_name = 'multiple_dogs.jpg'
# img_name = 'snake.JPEG'
img_name = "Bulk_Carrier_100.bmp"
img_path = os.path.join(img_dir, img_name)

pil_img = PIL.Image.open(img_path)

# normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
torch_img = (
    torch.from_numpy(np.asarray(pil_img))
    .permute(2, 0, 1)
    .unsqueeze(0)
    .float()
    .div(255)
    .cuda()
)
torch_img = F.upsample(torch_img, size=(224, 224), mode="bilinear", align_corners=False)
normed_torch_img = torch_img  # normalizer(torch_img)

resnet = torch.load("mbr_best_model.pth")
resnet.eval(), resnet.cuda()


cam_dict = dict()

resnet_model_dict = dict(
    type="resnet", arch=resnet, layer_name="layer4", input_size=(224, 224)
)
resnet_gradcam = GradCAM(resnet_model_dict, True)
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
cam_dict["resnet"] = [resnet_gradcam, resnet_gradcampp]

images = []
for gradcam, gradcam_pp in cam_dict.values():
    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

    # images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
    images.append(torch.stack([torch_img.squeeze().cpu(), result_pp], 0))

images = make_grid(torch.cat(images, 0), nrow=5)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_name = img_dir.split("/")[-2] + "_" + img_name
output_path = os.path.join(output_dir, output_name)

save_image(images, output_path)
PIL.Image.open(output_path)

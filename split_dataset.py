import os
import random


root = "/home/xjw/jianwen/data/ShipRSImageNet_patches/obb_patch2/"
train_file_path = root + "train.txt"
test_file_path = root + "test.txt"

if os.path.exists(train_file_path):
    os.remove(train_file_path)
    os.remove(test_file_path)

dirs = os.listdir(root)
train_per = 0.8

train_file = open(train_file_path, "a")
test_file = open(test_file_path, "a")

cls = -1
for idx, dir in enumerate(dirs):
    if os.path.isdir(os.path.join(root, dir)):
        cls += 1
        imgs = os.listdir(os.path.join(root, dir))
        img_path = [dir + "/" + i for i in imgs]
        num_train = int(train_per * len(imgs))
        train_img = random.sample(img_path, num_train)
        test_img = list(set(img_path) - set(train_img))
        for t in range(len(train_img)):
            train_file.write(train_img[t] + " " + str(cls) + "\n")
        for te in range(len(test_img)):
            test_file.write(test_img[te] + " " + str(cls) + "\n")
        
train_file.close()
test_file.close()

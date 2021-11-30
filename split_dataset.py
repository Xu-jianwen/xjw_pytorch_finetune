import os
import random


root = "/home/xjw/jianwen/data/ShipRSImageNet/"
train_file_path = root + "train.txt"
val_file_path = root + "val.txt"
test_file_path = root + "test.txt"

if os.path.exists(train_file_path):
    os.remove(train_file_path)
    os.remove(val_file_path)
    os.remove(test_file_path)

dirs = os.listdir(root)
train_per = 0.8
val_per = 0.1

train_file = open(train_file_path, "a")
val_file = open(val_file_path, "a")
test_file = open(test_file_path, "a")

cls = -1
for idx, dir in enumerate(dirs):
    if os.path.isdir(os.path.join(root, dir)):
        cls += 1
        imgs = os.listdir(os.path.join(root, dir))
        img_path = [dir + "/" + i for i in imgs]
        num_train = int(train_per * len(imgs))
        train_img = random.sample(img_path, num_train)
        val_and_test = list(set(img_path) - set(train_img))
        val_img = random.sample(val_and_test, int(len(val_and_test)*(val_per/(1-train_per))))
        test_img = list(set(img_path) - set(train_img) - set(val_img))
        for t in range(len(train_img)):
            train_file.write(train_img[t] + "," + str(cls) + "\n")
        for v in range(len(val_img)):
            val_file.write(val_img[v] + "," + str(cls) + "\n")
        for te in range(len(test_img)):
            test_file.write(test_img[te] + "," + str(cls) + "\n")
        
train_file.close()
test_file.close()

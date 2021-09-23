from torchvision import datasets, transforms
import torch
import multiprocessing
multiprocessing.set_start_method('spawn',True)

def load_data_train(root_path, dir, batch_size, kwargs):
    transform_dict = transforms.Compose(
        [transforms.Resize([224, 224]),
         # transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader

def load_data_val(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,pin_memory=True)
    return val_loader
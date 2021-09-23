from torchvision import datasets, transforms
import torch
import multiprocessing
multiprocessing.set_start_method('spawn',True)


def load_data_train(root_path, dir, batch_size, kwargs):
    transform_dict = transforms.Compose(
        [transforms.Resize([256, 256]), 
         transforms.RandomCrop(224), 
         transforms.RandomHorizontalFlip(), 
         transforms.ToTensor(), 
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader

def load_data_test(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(), 
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,pin_memory=True)
    return test_loader

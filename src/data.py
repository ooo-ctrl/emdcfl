import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform, color=False):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        self.color = color

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], int(self.labels[idx])
        if not self.color:
            image = Image.fromarray(image, mode='L')
        else:
            image = Image.fromarray(image, mode='RGB')
        image = self.transform(image)
        return image, label

class CSVDataset(Dataset):
    def __init__(self, metadata, transform):
        super().__init__()
        self.metadata_df = pd.read_csv(metadata)
        self.labels = self.metadata_df['label'].values
        self.image_paths = self.metadata_df['image_path'].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label

class DatasetWithIndices(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        image, label = self.dataset.__getitem__(idx)
        return image, label, idx
    
def get_transform(data='MNIST', split=None):

    if data == 'MNIST':
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.1307,), (0.3081,))
            ]
        )
    elif data =="IN":
        transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))                                
            ]
        )
    elif data =="CMNIST":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))                                
            ]
        )
    else: 
        print("Data choice invalid, exiting...")
        exit(1)

    return transform


def load_rotmnistmax(case, split, center):

    path = f"data/MNIST/rotmnistmax{case}c/rotmnist_{split}_{center}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("MNIST"))


def load_rotmnistno(case, split, center):

    path = f"data/MNIST/rotmnistno{case}c/rotmnist_{split}_{center}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("MNIST"))


def load_rotmnistmin(case, split, center):

    path = f"data/MNIST/rotmnistmin{case}c/rotmnist_{split}_{center}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("MNIST"))


def load_rotmnisttwo(case, split, center):

    path = f"data/MNIST/rotmnisttwo{case}c/rotmnist_{split}_{center}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("MNIST"))


def load_cmnist3x(case, split, center, clean):

    if clean:
        path = f"data/MNIST/cmnist3x{case}c_clean/cmnist_{split}_{center}.npz"
    else:
        path = f"data/MNIST/cmnist3x{case}c/cmnist_{split}_{center}.npz"

    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("CMNIST"), color=True)


def load_cifar10ls(split, center, clean):

    if clean:
        path = f"data/CIFAR10/CIFAR10-LS_clean/cifar10-ls_{split}_{center}.npz"
    else:
        path = f"data/CIFAR10/CIFAR10-LS/cifar10-ls_{split}_{center}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("IN", split), color=True)


def load_rotcifar10hardmax(case, split, center):

    path = f"data/CIFAR10/rotcifar10hardmax{case}c/rotcifar10_{split}_{center}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("IN", split), color=True)

def load_rotcifar10hardmin(case, split, center):

    path = f"data/CIFAR10/rotcifar10hardmin{case}c/rotcifar10_{split}_{center}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("IN", split), color=True)


def load_rotcifar10hardno(case, split, center):

    path = f"data/CIFAR10/rotcifar10hardno{case}c/rotcifar10_{split}_{center}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("IN", split), color=True)

def load_rotcifar10hardtwo(case, split, center):

    path = f"data/CIFAR10/rotcifar10hardtwo{case}c/rotcifar10_{split}_{center}.npz"
    data = np.load(path)
    print("loaded from", path)

    return ImageDataset(data['images'], data['labels'], transform=get_transform("IN", split), color=True)


def load_pacs(case, split, center):

    if case == '2':
        path = f"data/PACS/pacs_{split}_{center}.csv"
    else:
        path = f"data/PACS/pacs_{case}_{split}_{center}.csv"
    print("loaded metadata from", path)

    return CSVDataset(metadata=path, transform=get_transform("IN", split))



def load_labels(data):
    if 'rotmnist' in data:
        return [i for i in range(10)]
    elif 'cifar10' in data:
        return list(range(10))
    elif 'pacs' in data:
        return list(range(7))
    elif 'cmnist' in data:
        return [i for i in range(10)]
    else:
        print("Unsupported data, exiting...")
        exit(1) 


def load_data(data, split, center):

    print("Loading", data, split, "Center", center)

    if data == 'rotmnistmax10c':
        return load_rotmnistmax(case=10, split=split, center=center)
    elif data == 'rotmnistno10c':
        return load_rotmnistno(case=10, split=split, center=center)
    elif data == 'rotmnistmin10c':
        return load_rotmnistmin(case=10, split=split, center=center)
    elif data == 'rotmnisttwo10c':
        return load_rotmnisttwo(case=10, split=split, center=center)

    elif data == 'cmnist3x10c':
        return load_cmnist3x(case=10, split=split, center=center, clean=False)
    elif data == 'cmnist3x10c_clean':
        return load_cmnist3x(case=10, split=split, center=center, clean=True)

    elif data == 'cifar10-ls':
        return load_cifar10ls(split=split, center=center, clean=False)
    elif data == 'cifar10-ls_clean':
        return load_cifar10ls(split=split, center=center, clean=True)

    elif data == 'rotcifar10hardmax10c':
        return load_rotcifar10hardmax(case=10, split=split, center=center)
    elif data == 'rotcifar10hardmin10c':
        return load_rotcifar10hardmin(case=10, split=split, center=center)
    elif data == 'rotcifar10hardno10c':
        return load_rotcifar10hardno(case=10, split=split, center=center)
    elif data == 'rotcifar10hardtwo10c':
        return load_rotcifar10hardtwo(case=10, split=split, center=center)

    elif data == 'pacs500':
        return load_pacs(case=500, split=split, center=center)

    else:
        print("Unsupported data, exiting...")
        exit(1)
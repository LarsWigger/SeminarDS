'''
Based on https://www.kaggle.com/nachiket273/cyclegan-pytorch
'''
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

data_dir = "/home/monet/data"

class RandomPhotoAndMonetDataset(Dataset):
    def __init__(self, size=(256, 256), normalize=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__()
        self.monet_dir = os.path.join(data_dir, "monet_jpg")
        self.photo_dir = os.path.join(data_dir, "photo_jpg")
        self.monet_filenames = sorted(os.listdir(self.monet_dir))
        self.photo_filenames = sorted(os.listdir(self.photo_dir))
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)                                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()                               
            ])

    def __getitem__(self, idx):
        '''Returns a tuple of a random photo and a chosen monet, both transformed in the same way'''
        rand_idx = int(np.random.uniform(0, len(self.photo_filenames)))
        photo_path = os.path.join(self.photo_dir, self.photo_filenames[rand_idx])
        monet_path = os.path.join(self.monet_dir, self.monet_filenames[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        monet_img = Image.open(monet_path)
        monet_img = self.transform(monet_img)
        return (photo_img, monet_img)

    def __len__(self):
        return len(self.monet_filenames)
    
class PhotoDataset(Dataset):
    # returns only a photo as specified by the index
    def __init__(self, size=(256, 256), normalize=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__()
        self.photo_dir = os.path.join(data_dir, "photo_jpg")
        self.photo_filenames = self.photo_filenames = sorted(os.listdir(self.photo_dir))
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)                                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()                               
            ])

    def __getitem__(self, idx):
        photo_path = os.path.join(self.photo_dir, self.photo_filenames[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        return photo_img

    def __len__(self):
        return len(self.photo_filenames)

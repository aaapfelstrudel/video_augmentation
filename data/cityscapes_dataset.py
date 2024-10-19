import os
import sys
import os.path as osp
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image

class cityscapesDataSet(data.Dataset):

    def __init__(self, root, color='RGB', city='berlin', resize=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.root = root  # Folder for GTA5 which contains subfolder images
        self.color = color
        self.resize = resize
        self.mean = mean
        self.std = std
        self.status = False

        # Get all subfolders in the root directory
        self.existing_files = city
        self.frames = []

    def is_empty(self, lst):
        return len(lst) == 0

    def find_target(self, lst, target):
        return lst.index(target)

    def compilation(self, folder_name):
        # Get all frames from the specified folder
        folder_path = osp.join(self.root, folder_name)
        frames = os.listdir(folder_path)
        return frames

    def data_transform(self, image):
        # Define the transformation pipeline for image processing
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std) if self.color == 'RGB' else transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return transform_pipeline(image)        
        

    def __getitem__(self):

        if self.is_empty(self.frames):
            # Get the folder name corresponding to the given index
            folder_name = self.existing_files
            self.frames = self.compilation(folder_name)

        # Get the current frame name
        frame_name = self.frames.pop(0)

        # Build the full path to the image file
        folder_name = self.existing_files
        img_path = osp.join(self.root, folder_name, frame_name)
        # print(self.existing_files)

        # Load and transform the image
        image = Image.open(img_path).convert(self.color)
        image = self.data_transform(image)

        # Create a label indicating the source folder and frame name
        label = f"{frame_name}"
        if self.is_empty(self.frames):
            print(f"{self.existing_files} Dataset fully traversed.")
            self.status = True

        return image, label, self.status


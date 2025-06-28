# live_dataset_with_depth.py

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class IQADataset(Dataset):
    def __init__(self, dataset_loader, index, resize=None, aspect_ratio_size=None):
        self.images, self.scores = dataset_loader.load_dataset(index)

        if resize:
            self.resize = resize
            self.resize_transform = transforms.Resize(resize)
        else:
            self.resize = None

        self.aspect_ratio_size = aspect_ratio_size
        
        self.transform = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.290, 0.295, 0.300])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        score = self.scores[idx]
        image = Image.open(self.images[idx]).convert("RGB")

        if self.resize:
            image = self.resize_transform(image)

        if self.aspect_ratio_size:
            width, height = image.size
            if min(width, height) > self.aspect_ratio_size:
                # Resize the shortest side to `min_size` while keeping the aspect ratio.
                scale = self.aspect_ratio_size / min(width, height)
                new_width = int(round(width * scale))
                new_height = int(round(height * scale))
                image = image.resize((new_width, new_height), Image.BICUBIC)

        image = self.transform(image)
        if image is None:
            raise ValueError(f"Failed to load image or depth map for index {idx}")
    
        image_full = self.normalize(image)
        return image_full, torch.tensor(score, dtype=torch.float32)

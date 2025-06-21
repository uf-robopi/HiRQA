import os
import time
import torch
import numpy as np
import csv
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from model.HiRQA_similarity_backbone_avgpool import HiRQA_pretrain,HiRQA
from torchvision import transforms

def evaluate(model, image_paths, device):
    model.eval()
    predictions = []
    val_time = 0
 
    with torch.no_grad():
            normalize = transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.290, 0.295, 0.300])
            img = Image.open(image_path).convert("RGB")

            # Preprocess the images
            img = transforms.ToTensor()(img)
            img = normalize(img).unsqueeze(0).to(device)
            start_time = time.time()

            outputs = model(img)  # Get model predictions

            val_time += time.time() - start_time
            predictions.extend(outputs.cpu().numpy())  # Save prediction

    predictions = np.array(predictions).squeeze()
    fps = val_time / len(image_paths)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate IQA scores for images in a folder.')
    parser.add_argument('--img_path', type=str, required=True, help='Image Path')   
    parser.add_argument('--model', type=str, default="Resnet", help='model choice[HiRQA/HiRQA-S]')
    parser.add_argument("--ckpt", type=str, default="", help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HiRQA(model=args.model).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")["model_state_dict"])

    normalize = transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.290, 0.295, 0.300])
    img = Image.open(args.img_path).convert("RGB")
    img = transforms.ToTensor()(img)
    img = normalize(img).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(img).cpu().numpy()[0]
    
    print(f"The predicted quality score is {score:3f}")
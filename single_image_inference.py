import torch
from PIL import Image
import argparse
from model.HiRQA import HiRQA
from torchvision import transforms
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate IQA scores for images in a folder.')
    parser.add_argument('--img_path', type=str, required=True, help='Image Path')   
    parser.add_argument('--model', type=str, default="Resnet", help='model choice[HiRQA/HiRQA-S]')
    parser.add_argument("--ckpt", type=str, default="", help="Path to model weights")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HiRQA(model=args.model).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")["model_state_dict"])

    normalize = transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.290, 0.295, 0.300])
    img = Image.open(args.img_path).convert("RGB")
    img = transforms.ToTensor()(img)
    img = normalize(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(img)
    print(f"The predicted quality score is {score.cpu().numpy()[0] :.3f}")
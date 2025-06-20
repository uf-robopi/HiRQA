import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import numpy as np
from model.HiRQA import HiRQA
from dataset import folders 
from dataset.iqa_dataset import IQADataset
import argparse

def safe_correlation(predictions, targets):
    if np.allclose(predictions, predictions[0]) or np.allclose(targets, targets[0]):
        return 0.0
    return pearsonr(predictions, targets)[0]

def safe_spearman(predictions, targets):
    if np.allclose(predictions, predictions[0]) or np.allclose(targets, targets[0]):
        return 0.0
    return spearmanr(predictions, targets)[0]   

def evaluate(model, val_loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, scores in tqdm(val_loader):
            images = images.to(device)
            scores = scores.to(device)
            outputs = model(images)
            outputs = outputs.view(-1) 

            predictions.extend(outputs.cpu().numpy())
            targets.extend(scores.cpu().numpy())
        
    predictions = np.array(predictions).squeeze()
    targets = np.array(targets)

    plcc = safe_correlation(predictions, targets)
    srcc = safe_spearman(predictions, targets)
    
    return plcc, srcc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HiRQA evaluation.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--model', type=str, default="HiRQA", choices=['HiRQA','HiRQA-S'], help='Model version [HiRQA/HiRQA-S]')
    parser.add_argument("--ckpt", type=str, default="", help="Path to model weights")
    args = parser.parse_args()

    # model definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HiRQA(model=args.model).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")["model_state_dict"])
    
    # dataset loader
    dataset = args.dataset
    dataset_root = args.dataset_path
    resize = None
    aspect_ratio_size = None

    if dataset == 'csiq':
        dataset_loader = folders.CSIQ_loader(dataset_root)
    elif dataset == 'kadid_10k':
        dataset_loader = folders.KADID10k_loader(dataset_root)
    elif dataset == 'koniq':
        dataset_loader = folders.KONIQ_loader(dataset_root)
    elif dataset == 'live':
        dataset_loader = folders.LIVE_loader(dataset_root)
    elif dataset == 'tid_2013':
        dataset_loader = folders.TID_loader(dataset_root)
    elif dataset == 'livec':
        dataset_loader = folders.LIVECHALLENGE_loader(dataset_root)
    elif dataset == 'flive':
        resize = (512,512)
        dataset_loader = folders.FLIVE_loader(dataset_root)
    elif dataset == 'spaq':
        aspect_ratio_size = 512
        dataset_loader = folders.SPAQ_loader(dataset_root)
    else:
        print(f"Dataset {dataset} not recognized.")

    index = list(range(len(dataset_loader.ref_img_names)))
    dataset = IQADataset(dataset_loader, index, resize=resize, aspect_ratio_size=aspect_ratio_size)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 4)
    
    # evaluation
    plcc, srcc = evaluate(model, loader, device)
    print(f" |  SRCC: {srcc:.3f} |  PLCC: {plcc:.3f}")






    


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from setup.config import Config
from utils.metrics import KeyPointMetric


# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for _, (images, target) in enumerate(pbar):
        images = images.to(device)
        heatmaps = target['heatmaps'].to(device)

        # Forward pass
        pred_heatmaps = model(images)
        loss = criterion(pred_heatmaps, heatmaps)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    gt_keypts_list, gt_area_list = [], []
    pred_kpts_list, pred_conf_list = [], []

    pbar = tqdm(dataloader, desc='Evaluating')

    with torch.no_grad():
        for _, (images, target) in enumerate(pbar):
            images = images.to(device)

            if not isinstance(target, dict):
                 raise ValueError('Target must be a dictionary')

            heatmaps = target['heatmaps'].to(device)

            # Forward pass
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, heatmaps)
            total_loss += loss.item()

            #Add the results
            gt_keypts_list.append(target['keypoints'].detach().cpu())   # (B, K, 3) torch
            gt_area_list.append(target['seg_area'].detach().cpu())    # (B,) torch

            pred_keypoints = heatmap_to_keypoints(pred_heatmaps) 
            pred_conf = heatmap_to_conf(pred_heatmaps)
            pred_kpts_list.append(torch.from_numpy(pred_keypoints))    # (B, K, 3) torch
            pred_conf_list.append(pred_conf.detach().cpu())       
    
    
    if len(pred_kpts_list) == 0:
        return {"loss" : total_loss / len(dataloader), 
                "AP@0.50": 0.0, 
                "precision": 0.0, 
                "recall": 0.0}
     
    # after loop
    gt_kpts  = torch.cat(gt_keypts_list,  dim=0)  # (N, K, 3)
    gt_areas = torch.cat(gt_area_list, dim=0)  # (N,)
    pred_kpts  = torch.cat(pred_kpts_list,  dim=0)  # (N, K, 3)
    pred_conf  = torch.cat(pred_conf_list,  dim=0)  # (N,)

    #metrics_device = torch.device("cpu") # Causing GPU explode

    metric = KeyPointMetric(
                device=device,
                gt_kpts=gt_kpts,
                gt_areas=gt_areas,
                pred_kpts=pred_kpts,
                pred_conf=pred_conf)
    
    # Debug info about which mode was used
    mode_info = metric.get_mode_info()
    print(f"[Metrics] Mode: {mode_info['mode']}, GT: {mode_info['gt_count']}, Pred: {mode_info['pred_count']}")
    
    metric.threshold = 0.50

    precision, recall = metric.compute_pr()
    ap = metric.average_precision(precision, recall)

    precision_scalar = precision[-1].item()  # Last value (highest confidence)
    recall_scalar = recall[-1].item()        # Last value (highest confidence)

    print(f"Loss = {total_loss/len(dataloader)} |[eval] AP@0.50: {ap:.4f}")

    return {"loss": total_loss / len(dataloader), 
            "AP@0.50": ap, 
            "precision": precision_scalar,
            "recall": recall_scalar}


def heatmap_to_conf(heatmaps):
    """
    heatmaps: (P, K, H, W) logits or probs (if logits, we apply sigmoid)
    """
    probs = torch.sigmoid(heatmaps)
    peak_per_kpt, _ = probs.view(probs.size(0), probs.size(1), -1).max(dim=-1)  
    return peak_per_kpt.mean(dim=1)

def heatmap_to_keypoints(heatmaps):
    """Convert heatmaps to keypoint coordinates"""
    batch_size, num_keypoints, h, w = heatmaps.shape
    keypoints = np.zeros((batch_size, num_keypoints, 3))

    heatmaps_np = heatmaps.cpu().numpy()

    for b in range(batch_size):
        for k in range(num_keypoints):
            heatmap = heatmaps_np[b, k]

            # Find maximum
            idx = np.argmax(heatmap)
            y, x = np.unravel_index(idx, heatmap.shape)

            # Scale back to image size
            x = x * Config.img_size / Config.heatmap_size
            y = y * Config.img_size / Config.heatmap_size

            # Confidence is max value
            conf = heatmap.max()

            keypoints[b, k] = [x, y, conf]

    return keypoints

def calculate_pck(pred_keypoints, gt_keypoints, threshold=0.2):
    """Calculate PCK@0.2 metric"""
    distances = np.sqrt(np.sum((pred_keypoints[:, :, :2] - gt_keypoints[:, :, :2])**2, axis=2))

    # Normalize by person size (max distance between any two keypoints)
    person_sizes = []
    for gt in gt_keypoints:
        visible = gt[:, 2] > 0
        if visible.sum() > 1:
            visible_kpts = gt[visible, :2]
            dists = np.sqrt(np.sum((visible_kpts[:, None] - visible_kpts[None, :])**2, axis=2))
            person_sizes.append(dists.max())
        else:
            person_sizes.append(Config.img_size)

    person_sizes = np.array(person_sizes)[:, None]
    normalized_dists = distances / (person_sizes + 1e-6)

    # Calculate PCK
    visible_mask = gt_keypoints[:, :, 2] > 0
    correct = (normalized_dists < threshold) & visible_mask
    pck = correct.sum() / visible_mask.sum()

    return pck
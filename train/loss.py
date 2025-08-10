import torch
import torch.nn as nn
import torch.nn.functional as F

# The scope of this file is to implement the loss functions 
# for the keypoint detection task


class KeypointFocalLoss(nn.Module):
    """
    Focal loss for keypoint detection
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(KeypointFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred_heatmaps, gt_heatmaps):
        # Ensure predictions are in [0, 1] range
        if pred_heatmaps.max() > 1.0:
            pred_heatmaps = torch.sigmoid(pred_heatmaps)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(pred_heatmaps, gt_heatmaps, reduction='none')
        
        # Focal term
        pt = torch.where(gt_heatmaps == 1, pred_heatmaps, 1 - pred_heatmaps)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_weight = torch.where(gt_heatmaps == 1, self.alpha, 1 - self.alpha)
        
        # Final focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply mask for visible keypoints
        mask = (gt_heatmaps.sum(dim=(2, 3), keepdim=True) > 0).float()
        focal_loss = (focal_loss * mask).sum() / (mask.sum() + 1e-6)
        
        return focal_loss

# Original MSE loss
class KeypointMSELoss(nn.Module):
    def __init__(self):
        super(KeypointMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_heatmaps, gt_heatmaps):
        # Compute MSE loss with visibility mask
        loss = self.mse(pred_heatmaps, gt_heatmaps)

        # Apply mask for visible keypoints (gt_heatmap > 0)
        mask = (gt_heatmaps.sum(dim=(2, 3), keepdim=True) > 0).float()
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)

        return loss

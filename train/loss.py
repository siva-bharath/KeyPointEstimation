import torch
import torch.nn as nn
import torch.nn.functional as F

# The scope of this file is to implement the loss functions 
# for the keypoint detection task


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

# Not tested with the current model
# Observed anomalies in the loss values
class KeypointFocalLoss(nn.Module):
    """
    Improved Focal Loss for keypoint detection using heatmaps
        """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', 
                 pos_weight=None, label_smoothing=0.0):
        super(KeypointFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, pred_heatmaps, gt_heatmaps):
        """
        Args:
            pred_heatmaps: Predicted heatmaps [B, C, H, W] where C is num_keypoints
            gt_heatmaps: Ground truth heatmaps [B, C, H, W]
        """
        # Ensure inputs are in the right format
        if pred_heatmaps.dim() != 4:
            raise ValueError(f"Expected 4D input, got {pred_heatmaps.dim()}D")
        
        # Create visibility mask for valid keypoints
        valid_mask = (gt_heatmaps.sum(dim=(2, 3), keepdim=True) > 0).float()
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            gt_heatmaps = gt_heatmaps * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Calculate binary cross entropy loss 
        bce_loss = F.binary_cross_entropy(
            pred_heatmaps, gt_heatmaps, reduction='none'
        )
        
        # Get probabilities for focal weight calculation (already probabilities)
        pred_probs = pred_heatmaps
        
        # Calculate focal loss components
        pt = torch.where(gt_heatmaps > 0, pred_probs, 1 - pred_probs)
        
        # Apply focal adjustment: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_weight = torch.where(gt_heatmaps > 0, self.alpha, 1 - self.alpha)
        
        # Combine all components
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply valid keypoint mask
        focal_loss = focal_loss * valid_mask
        
        # Apply reduction
        if self.reduction == 'mean':
            # Only average over valid keypoints
            valid_count = valid_mask.sum()
            if valid_count > 0:
                loss = focal_loss.sum() / valid_count
            else:
                loss = focal_loss.sum() * 0.0
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        elif self.reduction == 'none':
            loss = focal_loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        
        return loss

class KeypointFocalLossWithMSE(nn.Module):
    """
    Combined Focal Loss + MSE Loss for keypoint detection
    """
    def __init__(self, focal_weight=0.7, mse_weight=0.3, 
                 alpha=1.0, gamma=2.0, reduction='mean'):
        super(KeypointFocalLossWithMSE, self).__init__()
        self.focal_weight = focal_weight
        self.mse_weight = mse_weight
        self.focal_loss = KeypointFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.mse_loss = KeypointMSELoss()
        
    def forward(self, pred_heatmaps, gt_heatmaps):
        focal_loss = self.focal_loss(pred_heatmaps, gt_heatmaps)
        mse_loss = self.mse_loss(pred_heatmaps, gt_heatmaps)
        
        # Combine losses
        total_loss = self.focal_weight * focal_loss + self.mse_weight * mse_loss
        
        return total_loss


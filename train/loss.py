import torch
import torch.nn as nn

class KeypointLoss(nn.Module):
    def __init__(self):
        super(KeypointLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_heatmaps, gt_heatmaps):
        # Compute MSE loss with visibility mask
        loss = self.mse(pred_heatmaps, gt_heatmaps)

        # Apply mask for visible keypoints (gt_heatmap > 0)
        mask = (gt_heatmaps.sum(dim=(2, 3), keepdim=True) > 0).float()
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)

        return loss

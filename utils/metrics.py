import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt


# COCO keypoint sigmas for 17 keypoints
COCO_SIGMAS = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, 
                            .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0


class KeyPointMetric:
    def __init__(self,
                   device,
                   gt_kpts: torch.Tensor,
                   gt_areas: torch.Tensor,
                   pred_kpts: torch.Tensor,       # (P, 17, 3)
                   pred_conf: Optional[torch.Tensor] = None,  # (P,) per-instance score
                   sigmas: torch.Tensor = COCO_SIGMAS,
                   ):

        self.device = device
        self.gt_kpts  = gt_kpts.to(self.device)
        self.gt_areas = gt_areas.to(self.device)
        self.pred_kpts = pred_kpts.to(self.device)
        self.pred_conf = (
            pred_conf.to(self.device)
            if pred_conf is not None
            else torch.ones(len(pred_kpts), device=self.device)
        )
        self.sigmas = sigmas.to(self.device).float()

        self.threshold = 0.5 # oks confidence threshold
        self.num_gt = len(self.gt_kpts)
        self.num_pred = len(self.pred_kpts)

        # Auto-detect if we can use aligned mode
        self.can_use_aligned = (self.num_gt == self.num_pred)
        
        # Compute only once
        with torch.no_grad():
            if self.can_use_aligned:
                # Aligned mode: direct 1-to-1 comparison (fastest)
                vis_mask = (self.gt_kpts[..., 2] > 0).to(self.gt_kpts.dtype)
                dist_sq = (self.gt_kpts[..., 0] - self.pred_kpts[..., 0]) ** 2 + \
                          (self.gt_kpts[..., 1] - self.pred_kpts[..., 1]) ** 2
                denom = 2.0 * (self.sigmas ** 2)[None, :] * (self.gt_areas[:, None] + torch.finfo(torch.float32).eps)
                exp_term = dist_sq / denom
                oks_vec = (torch.exp(-exp_term) * vis_mask).sum(-1) / (vis_mask.sum(-1) + torch.finfo(torch.float32).eps)
                self._oks = oks_vec  # (N,) - per-sample OKS
                self._mode_used = "aligned"
            else:
                # Pairwise mode: handle different counts with smart filtering
                self._oks = self._compute_pairwise_oks()
                self._mode_used = "pairwise"

    def _compute_pairwise_oks(self):
        """Compute pairwise OKS matrix with smart filtering for efficiency"""
        # Compute full pairwise OKS matrix
        EPSILON = torch.finfo(torch.float32).eps
        
        dist_sq = (self.gt_kpts[:, None, :, 0] - self.pred_kpts[None, :, :, 0]) ** 2 + \
                  (self.gt_kpts[:, None, :, 1] - self.pred_kpts[None, :, :, 1]) ** 2
        
        vis_mask = (self.gt_kpts[..., 2] > 0).to(self.gt_kpts.dtype)
        
        areas_expanded = self.gt_areas[:, None, None]
        
        denom = 2.0 * (self.sigmas ** 2)[None, None, :] * (areas_expanded + EPSILON)
        
        exp_term = dist_sq / denom
        
        oks_matrix = (torch.exp(-exp_term) * vis_mask[:, None, :]).sum(-1) / \
                     (vis_mask[:, None, :].sum(-1) + EPSILON)
        
        return oks_matrix  
    
    @property
    def oks(self) -> torch.Tensor:
        return self._oks
    
    def compute_pr(self):
        if self.can_use_aligned:
            # Aligned case: one prediction per GT (fastest)
            return self._compute_pr_aligned()
        else:
            # Pairwise case: handle different counts with smart matching
            return self._compute_pr_pairwise()
    
    def _compute_pr_aligned(self):
        """Compute PR for aligned mode (1-to-1 mapping)"""
        P = self.oks.numel()
        sort_idx = torch.argsort(self.pred_conf, descending=True)
        oks_sorted = self.oks[sort_idx]

        tps = (oks_sorted >= self.threshold).to(torch.float32)
        fps = 1.0 - tps

        tpc = torch.cumsum(tps, 0)
        fpc = torch.cumsum(fps, 0)
        recall = tpc / max(self.num_gt, 1)
        precision = tpc / torch.clamp(tpc + fpc, min=1e-12)
        return precision, recall
    
    def _compute_pr_pairwise(self):
        """Compute PR for pairwise mode with smart matching"""
        G, P = self.oks.shape
        
        # Sort predictions by confidence (highest first)
        sort_idx = torch.argsort(self.pred_conf, descending=True)
        oks_sorted = self.oks[:, sort_idx]  # (G, P) - sorted by prediction confidence
        
        # Track which GTs and predictions have been matched
        gt_matched = torch.zeros(G, dtype=torch.bool, device=self.oks.device)
        pred_matched = torch.zeros(P, dtype=torch.bool, device=self.oks.device)
        
        # Initialize TP/FP arrays
        tps = torch.zeros(P, dtype=torch.float32, device=self.oks.device)
        fps = torch.zeros(P, dtype=torch.float32, device=self.oks.device)
        
        # Greedy matching: for each prediction (in confidence order), find best unmatched GT
        for pred_idx in range(P):
            if pred_matched[pred_idx]:
                continue
                
            # Get OKS scores for this prediction against all unmatched GTs
            pred_oks = oks_sorted[:, pred_idx]  # (G,) - OKS scores for this prediction
            pred_oks_masked = pred_oks.masked_fill(gt_matched, -1.0)  # Mask out matched GTs
            
            # Find best unmatched GT for this prediction
            best_gt_score, best_gt_idx = pred_oks_masked.max(dim=0)
            
            if best_gt_score >= self.threshold:
                # Match found: mark as TP
                tps[pred_idx] = 1.0
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True
            else:
                # No good match: mark as FP
                fps[pred_idx] = 1.0
                pred_matched[pred_idx] = True
        
        # Compute cumulative TP/FP
        tpc = torch.cumsum(tps, 0)
        fpc = torch.cumsum(fps, 0)
        
        # Compute precision and recall
        recall = tpc / max(self.num_gt, 1)
        precision = tpc / torch.clamp(tpc + fpc, min=1e-12)
        
        return precision, recall
    
    def average_precision(self, precision, recall):
        
        mrec = torch.cat([torch.tensor([0.], device=self.device), recall, torch.tensor([1.], device=self.device)])
        mpre = torch.cat([torch.tensor([0.], device=self.device), precision, torch.tensor([0.], device=self.device)])
        for i in range(mpre.numel() - 1, 0, -1):
            mpre[i-1] = torch.maximum(mpre[i-1], mpre[i])

        rec_thrs = torch.linspace(0, 1, 101, device=self.device)
        p_interp = torch.zeros_like(rec_thrs)
        for i, rt in enumerate(rec_thrs):
            inds = torch.nonzero(mrec >= rt, as_tuple=False).squeeze(-1)
            p_interp[i] = mpre[inds].max() if inds.numel() > 0 else 0.0

        return p_interp.mean().item()
    
    def get_mode_info(self):
        """Return information about which mode was used and why"""
        return {
            "mode": self._mode_used,
            "gt_count": self.num_gt,
            "pred_count": self.num_pred,
            "can_use_aligned": self.can_use_aligned,
            "oks_shape": list(self.oks.shape)
        }
    
    
if __name__ == '__main__':
    # Test example would go here
    pass
    
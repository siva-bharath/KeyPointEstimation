def compute_oks(gt_kpts: torch.Tensor, pred_kpts: torch.Tensor, 
                sigmas: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """
    KeyPoints similarity metric
    This compute is useful 
    when the instances predicted (N) > ground truth (M) or N < M 
    """
    EPSILON = torch.finfo(torch.float32).eps

    dist_sq = (gt_kpts[:, None, :, 0] - pred_kpts[..., 0]) ** 2 + (
        gt_kpts[:, None, :, 1] - pred_kpts[..., 1]) ** 2

    vis_mask = gt_kpts[..., 2].int() > 0
    
    denom = 2 * (sigmas ** 2) * (areas[:, None, None] + EPSILON)

    exp_term = dist_sq / denom

    # Object Keypoint Similarity
    oks = (torch.exp(-exp_term) * vis_mask[:, None, :]).sum(-1) / (vis_mask[:, None, :].sum(-1) + EPSILON)

    return oks
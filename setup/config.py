from dataclasses import dataclass
from typing import Callable
import torch

@dataclass
class Config:
    '''
    Config for data
    '''
    data_dir: str = './dataset/coco_data'
    train_dir: str = 'train2017'
    valid_dir: str = 'val2017'
    train_ann_file: str = 'annotations/person_keypoints_train2017.json'
    valid_ann_file: str = 'annotations/person_keypoints_val2017.json'

    # Model Config
    num_keypoints: int = 17
    img_size: int = 256
    heatmap_size: int = 64
    sigma: float = 2.0
    
    # Epoch Config
    optimizer: torch.optim.Optimizer = None
    scheduler: Callable = None

    # Training Config
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.0001  # Reduced from 0.001 for better convergence
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_interval: int = 5
    early_stopping_patience: int = 10
    
    # Loss Config
    loss_type: str = 'mse'  # Options: 'focal', 'mse', 'combined'
    focal_alpha: float = 1.0  # Reduced from 2.0 for better balance
    focal_gamma: float = 2.0  # Focus on hard examples
    label_smoothing: float = 0.1  
    focal_weight: float = 0.7  
    mse_weight: float = 0.3   
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from setup.config import Config

import cv2
from pycocotools.coco import COCO
 
# Dataset class
class COCOKeypointDataset(Dataset):
    def __init__(self, config: Config, transform=None, is_train=True):
        """       
        config : Configuration
        """
        self.config = config
        self.is_train = is_train
        
        if is_train: 
            image_dir = config.train_dir 
            annotation_file = config.train_ann_file
        else: 
            image_dir = config.valid_dir
            annotation_file = config.valid_ann_file

        self.img_dir = os.path.join(config.data_dir, image_dir)
        self.coco = COCO(os.path.join(config.data_dir, annotation_file))
        self.transform = transform
        
        # Get all person annotations with keypoints
        self.ids = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=0)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if ann['num_keypoints'] > 0:
                    self.ids.append(ann['id'])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.loadAnns([ann_id])[0]
        img_info = self.coco.loadImgs(ann['image_id'])[0]

        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get bounding box and keypoints
        bbox = ann['bbox']  # [x, y, width, height]
        area = float(ann['area'])
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)  # [x, y, visibility]

        # Crop person from image
        x, y, w, h = [int(v) for v in bbox]
        # Add padding
        pad = int(max(w, h) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        person_img = img[y1:y2, x1:x2]

        # Adjust keypoints to cropped image
        keypoints[:, 0] -= x1
        keypoints[:, 1] -= y1

        # Resize image and keypoints
        orig_h, orig_w = person_img.shape[:2]
        person_img = cv2.resize(person_img, (self.config.img_size, self.config.img_size))

        keypoints = keypoints.astype(np.float32)

        scale_x = self.config.img_size / orig_w
        scale_y = self.config.img_size / orig_h

        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        rescaled_area = area * scale_x * scale_y

        # Generate heatmaps
        heatmaps = self.generate_heatmaps(keypoints)

        # Convert to tensor
        if self.transform:
            person_img = self.transform(person_img)
        else:
            person_img = torch.FloatTensor(person_img).permute(2, 0, 1) / 255.0

        # defining the target
        target = {'heatmaps': torch.FloatTensor(heatmaps),  
                  'keypoints': torch.FloatTensor(keypoints),
                  'seg_area': torch.tensor(rescaled_area)}

        return person_img, target
    
    def generate_heatmaps(self, keypoints):
        """Generate gaussian heatmaps for keypoints"""
        heatmaps = np.zeros((self.config.num_keypoints, self.config.heatmap_size, self.config.heatmap_size))

        for i, (x, y, v) in enumerate(keypoints):
            if v > 0:  # visible
                # Scale to heatmap size
                x = x * self.config.heatmap_size / self.config.img_size
                y = y * self.config.heatmap_size / self.config.img_size

                # Generate gaussian
                heatmap = self.gaussian_2d(self.config.heatmap_size, self.config.heatmap_size,
                                         x, y, self.config.sigma)
                heatmaps[i] = heatmap

        return heatmaps

    def gaussian_2d(self, height, width, cx, cy, sigma):
        """Generate 2D gaussian heatmap"""
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]

        gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        return gaussian 
    
        
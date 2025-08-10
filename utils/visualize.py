# draw keypoints and visulize the skeleton 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image


def read_cv_image(image_path):
    """
    Read image from path and convert to cv2 format
    """
    image = cv2.imread(image_path)

    if image.empty():
        raise FileNotFoundError(f"Image empty found at {image_path}")

    elif image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return image

# create the list of keypoints.
human_keypoints = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow',
                'right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee', 'right_knee', 'left_ankle','right_ankle']


def get_limbs_from_keypoints(keypoints):
  limbs = [       
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
        ]
  return limbs

def draw_skeleton_per_person(img_path, output, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):    
    # Get the Inference
    limbs = get_limbs_from_keypoints(human_keypoints)
    img = read_cv_image(img_path)
    cmap = plt.get_cmap('rainbow')

    img_copy = img.copy()
    # check if the keypoints are detected
    if len(output["keypoints"])>0:
      # pick a set of N color-ids from the spectrum
      colors = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]
      for person_id in range(len(all_keypoints)):
          # check the confidence score of the detected person
          if confs[person_id]>conf_threshold:
            keypoints = all_keypoints[person_id, ...]

            # iterate for every limb 
            for limb_id in range(len(limbs)):
              limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().numpy().astype(np.int32)
              limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().numpy().astype(np.int32)
              limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
              if limb_score> keypoint_threshold:
                color = tuple(np.asarray(cmap(colors[person_id])[:-1])*255)
                cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), color, 25)

    return img_copy

def ensure_cv_format(image):
    """
    Convert various image formats to numpy array for OpenCV operations.   
    """
    # Handle PIL Image
    if isinstance(image, Image.Image):
        # PIL is RGB, convert to BGR for OpenCV
        image_np = np.array(image)
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Handle torch.Tensor
    elif isinstance(image, torch.Tensor):
        # Remove batch dimension if present
        if image.dim() == 4:  # [B, C, H, W]
            image = image.squeeze(0)
        
        # Convert from [C, H, W] to [H, W, C]
        if image.dim() == 3:
            image = image.permute(1, 2, 0)
        
        # Convert to numpy
        image_np = image.detach().cpu().numpy()
        
        # Handle different value ranges
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Handle numpy array
    elif isinstance(image, np.ndarray):
        # Ensure it's uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Ensure it's BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            # If it's RGB, convert to BGR
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL Image, torch.Tensor, or numpy.ndarray")

def draw_keypoints_per_person(image_path, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    """
    all_keypoints - [N, 17, 64, 64]
    """

    # Consts
    CIRCLE_RADIUS = 30
    
    # Ensure image is in correct format for OpenCV
    if not isinstance(image_path, str):
        raise TypeError(f"image_path must be a string, got {type(image_path)}")

    img = read_cv_image(image_path)
    
    cmap = plt.get_cmap('rainbow')
    img_copy = img.copy()
    color_id = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]

    for person_id in range(len(all_keypoints)):
      if confs[person_id] > conf_threshold:
        keypoints = all_keypoints[person_id, ...]
        scores = all_scores[person_id, ...]
        for kp in range(len(scores)):
            if scores[kp]>keypoint_threshold:
                keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                cv2.circle(img_copy, keypoint, 30, color, -1)

    return img_copy


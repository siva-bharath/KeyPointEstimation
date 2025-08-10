import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# COCO keypoint connections for skeleton drawing
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], 
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
    [2, 4], [3, 5], [4, 6], [5, 7]
]

 
# COCO keypoint names for visualization
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO keypoint colors (BGR format for OpenCV)
COCO_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170)
]


class PoseNetInference:
    def __init__(self, onnx_path: str, img_size: int = 256, heatmap_size: int = 64):
        """
        Initialize PoseNet inference with ONNX model
        
        Args:
            onnx_path: Path to ONNX model file
            img_size: Input image size for the model
            heatmap_size: Output heatmap size from the model
        """
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.scale_factor = img_size / heatmap_size
        
        # Load ONNX model
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"ONNX model loaded from: {onnx_path}")
        print(f"Input shape: {self.session.get_inputs()[0].shape}")
        print(f"Output shape: {self.session.get_outputs()[0].shape}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            Preprocessed image tensor (1, C, H, W) in RGB format, normalized
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image_resized = cv2.resize(image_rgb, (self.img_size, self.img_size))
        
        # Convert to float and normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # Add batch dimension and transpose to (B, C, H, W)
        image_tensor = np.transpose(image_normalized, (2, 0, 1))[None, ...]
        
        return image_tensor
    
    def extract_keypoints(self, heatmaps: np.ndarray, confidence_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract keypoint coordinates and confidence from heatmaps
        
        Args:
            heatmaps: Model output heatmaps (K, H, W)
            confidence_threshold: Minimum confidence threshold for keypoints
            
        Returns:
            keypoints: (K, 3) array with [x, y, confidence] for each keypoint
            valid_mask: (K,) boolean mask indicating valid keypoints
        """
        num_keypoints = heatmaps.shape[0]
        keypoints = np.zeros((num_keypoints, 3))
        valid_mask = np.zeros(num_keypoints, dtype=bool)
        
        for k in range(num_keypoints):
            heatmap = heatmaps[k]
            
            # Find maximum value and position
            max_val = np.max(heatmap)
            if max_val > confidence_threshold:
                # Get coordinates of maximum
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                # Scale back to image coordinates
                x_scaled = x * self.scale_factor
                y_scaled = y * self.scale_factor
                
                keypoints[k] = [x_scaled, y_scaled, max_val]
                valid_mask[k] = True
        
        return keypoints, valid_mask
    
    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, 
                     valid_mask: np.ndarray, draw_keypoints: bool = True, 
                     draw_skeleton: bool = True) -> np.ndarray:
        """
        Draw skeleton and keypoints on image
        
        Args:
            image: Input image (H, W, C) in BGR format
            keypoints: (K, 3) array with [x, y, confidence] for each keypoint
            valid_mask: (K,) boolean mask indicating valid keypoints
            draw_keypoints: Whether to draw individual keypoints
            draw_skeleton: Whether to draw skeleton connections
            
        Returns:
            Image with skeleton drawn
        """
        image_draw = image.copy()
        
        # Draw skeleton connections
        if draw_skeleton:
            for connection in COCO_SKELETON:
                kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1  # COCO indices are 1-based
                
                if valid_mask[kpt1_idx] and valid_mask[kpt2_idx]:
                    pt1 = tuple(keypoints[kpt1_idx][:2].astype(int))
                    pt2 = tuple(keypoints[kpt2_idx][:2].astype(int))
                    
                    # Draw line between keypoints
                    cv2.line(image_draw, pt1, pt2, (0, 255, 0), 2)
        
        # Draw individual keypoints
        if draw_keypoints:
            for k in range(len(keypoints)):
                if valid_mask[k]:
                    x, y, conf = keypoints[k]
                    x, y = int(x), int(y)
                    
                    # Draw circle for keypoint
                    color = COCO_COLORS[k]
                    cv2.circle(image_draw, (x, y), 5, color, -1)
                    
                    # Draw confidence text
                    cv2.putText(image_draw, f'{conf:.2f}', (x+10, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image_draw
    
    def predict(self, image: np.ndarray, confidence_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on image
        
        Args:
            image: Input image (H, W, C) in BGR format
            confidence_threshold: Minimum confidence threshold for keypoints
            
        Returns:
            keypoints: (K, 3) array with [x, y, confidence] for each keypoint
            valid_mask: (K,) boolean mask indicating valid keypoints
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        heatmaps = outputs[0][0]  # Remove batch dimension
        
        # Extract keypoints
        keypoints, valid_mask = self.extract_keypoints(heatmaps, confidence_threshold)
        
        return keypoints, valid_mask
    
    def visualize_prediction(self, image: np.ndarray, keypoints: np.ndarray, 
                           valid_mask: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize prediction results
        
        Args:
            image: Input image
            keypoints: Extracted keypoints
            valid_mask: Valid keypoint mask
            save_path: Optional path to save visualization
            
        Returns:
            Image with visualization
        """
        # Draw skeleton
        result_image = self.draw_skeleton(image, keypoints, valid_mask)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"Visualization saved to: {save_path}")
        
        return result_image


def main():
    parser = argparse.ArgumentParser(description='PoseNet ONNX Inference')
    parser.add_argument('--model', type=str, default='checkpoints/posenet_model.onnx',
                       help='Path to ONNX model file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='Path to save output image')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for keypoints')
    parser.add_argument('--img-size', type=int, default=256,
                       help='Input image size for model')
    parser.add_argument('--heatmap-size', type=int, default=64,
                       help='Output heatmap size from model')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: ONNX model not found at {args.model}")
        print("Please train the model first or check the path.")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    # Initialize inference
    try:
        pose_net = PoseNetInference(args.model, args.img_size, args.heatmap_size)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    print(f"Processing image: {args.image}")
    print(f"Image shape: {image.shape}")
    
    # Run inference
    try:
        keypoints, valid_mask = pose_net.predict(image, args.confidence)
        
        # Print results
        num_valid = np.sum(valid_mask)
        print(f"Detected {num_valid}/{len(keypoints)} keypoints")
        
        for i, (kpt, valid) in enumerate(zip(keypoints, valid_mask)):
            if valid:
                print(f"{COCO_KEYPOINT_NAMES[i]}: ({kpt[0]:.1f}, {kpt[1]:.1f}), conf: {kpt[2]:.3f}")
        
        # Visualize and save
        result_image = pose_net.visualize_prediction(image, keypoints, valid_mask, args.output)
        
        # Display image (optional)
        cv2.imshow('PoseNet Prediction', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == '__main__':
    main()

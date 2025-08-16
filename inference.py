import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os

from utils.visualize import draw_skeleton_per_person

model_to_coco = None

class PoseNetONNX:
    def __init__(self, onnx_path, img_size=256, heatmap_size=64):
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.scale_factor = img_size / heatmap_size

        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.img_size, self.img_size))
        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_normalized - mean) / std
        image_tensor = np.transpose(image_normalized, (2, 0, 1))[None, ...]
        return image_tensor.astype(np.float32)

    def extract_keypoints(self, heatmaps, confidence_threshold=0.0):
        num_keypoints = heatmaps.shape[0]
        keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)
        valid_mask = np.zeros(num_keypoints, dtype=bool)

        for k in range(num_keypoints):
            hm = heatmaps[k]
            max_val = np.max(hm)
            if max_val > confidence_threshold:
                y, x = np.unravel_index(np.argmax(hm), hm.shape)
                keypoints[k, 0] = x
                keypoints[k, 1] = y
                keypoints[k, 2] = max_val
                valid_mask[k] = True

        return keypoints, valid_mask

    def predict(self, image, confidence_threshold=0.3):
        H0, W0 = image.shape[:2]
        input_tensor = self.preprocess_image(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        heatmaps = outputs[0][0]  # (K,H,W)

        # decode keypoints in heatmap coords
        keypoints_hm, valid_mask = self.extract_keypoints(heatmaps, confidence_threshold)

        # map heatmap coords -> model input -> original image
        sx = self.img_size / self.heatmap_size
        sy = self.img_size / self.heatmap_size
        keypoints_img = keypoints_hm.copy()
        keypoints_img[:, 0] *= sx * (W0 / self.img_size)
        keypoints_img[:, 1] *= sy * (H0 / self.img_size)

        # reorder if model_to_coco is defined
        if model_to_coco is not None:
            keypoints_img = keypoints_img[model_to_coco]
            valid_mask = valid_mask[model_to_coco]

        return keypoints_img, valid_mask


def main():
    parser = argparse.ArgumentParser(description='PoseNet ONNX Inference')
    parser.add_argument('--model', type=str, default='checkpoints/posenet_model.onnx',
                       help='Path to ONNX model file')
    parser.add_argument('--image', type=str, default='testimages/Image6.jpg',
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='testimages/output5.jpg',
                       help='Path to save output image')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for keypoints')
    parser.add_argument('--img-size', type=int, default=256,
                       help='Input image size for model')
    parser.add_argument('--heatmap-size', type=int, default=64,
                       help='Output heatmap size from model')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"ONNX model not found at {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found at {args.image}")

    image = cv2.imread(args.image)
    pose_net = PoseNetONNX(args.model, args.img_size, args.heatmap_size)
    keypoints, _ = pose_net.predict(image, args.confidence)

    # Prepare input signature for visualize.py 
    all_keypoints = np.expand_dims(keypoints, axis=0)  # (1,17,3)
    all_scores = np.expand_dims(keypoints[:, 2], axis=0)  # (1,17)
    confs = np.array([np.mean(all_scores)])  # (1,)

    # Output dict for 
    output_dict = {"keypoints": all_keypoints}
    
    # Draw skeleton 
    img_skel = draw_skeleton_per_person(args.image, 
        output_dict, all_keypoints, all_scores, confs,
        keypoint_threshold=0.2, conf_threshold=0.1
    )

    cv2.imwrite(args.output, img_skel)
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()

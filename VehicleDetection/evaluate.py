import torch
import cv2
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import warnings

class YOLOv5Evaluator:
    def __init__(self, model_path, test_dir):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.model_path = model_path
        self.test_dir = test_dir
        self.model = self.load_model()

    def load_model(self):
        """Load trained YOLOv5 model."""
        return torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, source='github')

    def visualize_training_results(self, results_path):
        """Plot training and validation metrics."""
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()

        # Loss Graph
        plt.figure(figsize=(14, 6))
        plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
        plt.plot(df['epoch'], df['train/obj_loss'], label='Train Obj Loss')
        plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
        plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linestyle='--')
        plt.plot(df['epoch'], df['val/obj_loss'], label='Val Obj Loss', linestyle='--')
        plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('📉 Training & Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Metrics Graph
        plt.figure(figsize=(14, 6))
        plt.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
        plt.plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
        plt.plot(df['epoch'], df['metrics/precision'], label='Precision')
        plt.plot(df['epoch'], df['metrics/recall'], label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Score')
        plt.title('📈 Validation Metrics per Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

    def infer_on_test_images(self, num_samples=5):
        """Run inference on test images and display results."""
        sample_imgs = random.sample(os.listdir(self.test_dir), num_samples)
        plt.figure(figsize=(15, 10))
        for i, img_name in enumerate(sample_imgs):
            img_path = os.path.join(self.test_dir, img_name)
            results = self.model(img_path)
            results.render()
            out_img = cv2.cvtColor(results.ims[0], cv2.COLOR_BGR2RGB)
            plt.subplot(2, 3, i + 1)
            plt.imshow(out_img)
            plt.axis('off')
            plt.title(img_name)
        plt.show()

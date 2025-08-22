import os
import subprocess

class YOLOv5Trainer:
    def __init__(self, working_dir, yolov5_dir='/kaggle/working/yolov5'):
        self.working_dir = working_dir
        self.yolov5_dir = yolov5_dir
        self.weights_url = 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l.pt'

    def setup_yolov5(self):
        """Clone YOLOv5 repository and install requirements."""
        os.chdir(self.working_dir)
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'], check=True)
        os.chdir(self.yolov5_dir)
        subprocess.run(['pip', 'install', '-qr', 'requirements.txt'], check=True)
        subprocess.run(['wget', self.weights_url], check=True)

    def train_model(self, yaml_path, img_size=640, epochs=15, batch_size=-1, name='veh_detect_l_fast'):
        """Train YOLOv5 model."""
        cmd = [
            'python', 'train.py',
            '--img', str(img_size),
            '--epochs', str(epochs),
            '--batch', str(batch_size),
            '--data', yaml_path,
            '--weights', 'yolov5l.pt',
            '--cache',
            '--name', name
        ]
        subprocess.run(cmd, check=True)
        return os.path.join(self.yolov5_dir, 'runs', 'train', name, 'weights', 'best.pt')

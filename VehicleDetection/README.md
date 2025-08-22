# Vehicle Detection using YOLOv5
## Objective

The objective of this project is to detect and classify vehicles (Car, Bus, Truck, Motorbike) in images using the Vehicle Detection Dataset.

Each image may contain zero or more objects, annotated in XML format. The task is to:

Convert annotations into YOLO format

Visualize data distribution (classes, bounding box width/height/area)

Train a YOLOv5 Object Detector

Evaluate performance using Precision, Recall, mAP

Provide results with plots and discussion

## Project Structure
Project-3-VehicleDetection/

│── data_preprocessing.py      # Convert XML → YOLO TXT

│── visualize_data.py          # Visualizations (class counts, bbox stats)

│── train.py                   # Training launcher script

│── evaluate.py                 # Run evaluation + plots

│── vehicle_data.yaml           # Dataset config file

│── VehicleDetection.ipynb      # Kaggle Notebook (main pipeline)

│── results/                    # Plots + trained weights + predictions

│── README.md                   # Documentation


# Workflow
## Step 1: Data Preparation

XML annotations converted to YOLO format (class x y w h)

Train/val/test split

## Step 2: Data Visualization

Class distribution histogram

Bounding box area, width, height scatter plots

Helps understand dataset balance

<img width="684" height="451" alt="download" src="https://github.com/user-attachments/assets/38b6efc2-98d3-42f4-9573-d0f20b637f26" />


## Step 3: Model Training (YOLOv5l)

Pretrained weights (yolov5l.pt)

Trained for 15 epochs on 640×640 images

Optimizer auto-configured by YOLOv5

## Step 4: Evaluation

Metrics: Precision (P), Recall (R), mAP@0.5, mAP@0.5:0.95

Per-class results reported

## Step 5: Results & Discussion

Save plots, confusion matrix, PR curves, sample predictions

## Data Visualization

Class Distribution
(Example output – saved in results/class_distribution.png)

Car → 506 objects

Bus → 266 objects

Truck → 124 objects

Motorbike → 179 objects

Bounding box area, width & height distributions plotted as histograms.

## Model Training
# Clone YOLOv5 repo
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -qr requirements.txt

### Download YOLOv5l pretrained weights
!wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l.pt

### Train YOLOv5l
!python train.py --img 640 --epochs 15 --batch -1 \
  --data /kaggle/working/vehicle-detection-dataset/vehicle_data.yaml \
  --weights yolov5l.pt --cache --name veh_detect_l_fast

## Evaluation Results

After 15 epochs, the model achieved:

Class	Precision (P)	Recall (R)	mAP@0.5	mAP@0.5:0.95
Car	0.871	0.630	0.798	0.569
Bus	0.874	0.726	0.826	0.603
Truck	0.830	0.677	0.792	0.577
Motorbike	0.891	0.640	0.742	0.366
All	0.866	0.668	0.789	0.529

<img width="662" height="278" alt="vc result" src="https://github.com/user-attachments/assets/1e9ebdf6-3874-40ec-b413-418d85857471" />


## Best Model Weights:

Saved at runs/train/veh_detect_l_fast/weights/best.pt

## Flowchart
flowchart TD
A[Input Images + XML Annotations] --> B[Convert XML → YOLO Format]
B --> C[Train/Val/Test Split]
C --> D[Data Visualization]
D --> E[YOLOv5 Training with Pretrained Weights]
E --> F[Evaluation Metrics (P, R, mAP)]
F --> G[Result Plots + Inference Samples]

## How to Run

Clone this repo inside Kaggle Notebook

Run data_preprocessing.py to generate YOLO labels

Run visualize_data.py to generate dataset stats

Run train.py (or Kaggle cells) for YOLOv5 training

Evaluate results using evaluate.py

## Sample Results

Predicted bounding boxes on test images:

## Discussion

The model performs best on Bus & Car detection due to higher sample counts

Lower recall for Trucks & Motorbikes → dataset imbalance

Future improvements:

Train for more epochs (50–100)

Use YOLOv8 or Faster R-CNN for comparison

Apply data augmentation (Mosaic, MixUp)

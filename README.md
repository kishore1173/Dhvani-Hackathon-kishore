 #Dhvani Hackathon 2025 Repository

### This repository contains three core projects developed for Dhvani Hackathon 2025, all solving separate tasks under the theme of computer vision and dynamics:

- **Defect Detection**
- **Vehicle Detection & Classification**
- **Bee Dynamics in 3D (Lorenz Attractor)**

--- 

##  Repository Structure

/
├── Defect Detection/

│ ├── detection.py

│ └── README.md

│

├── VehicleDetection/

│ ├── preprocess.py

│ ├── train.py

│ ├── evaluate.py

│ ├── visualize_data.py

│ ├── Vehicle Detection notebook.ipynb

│ └── README.md

│

├── bee-dynamics-3d/

│ ├── LorenzAttractor3D.py

│ ├── result.png

│ └── README.md

│

└── README.md (this file)


## You can clone your repo directly from GitHub with a single command.
Here’s the exact line you need to run in terminal / Kaggle / Colab:
```
git clone https://github.com/kishore1173/Dhvani-Hackathon-kishore.git
```

After cloning, move inside the repo:
```
cd Dhvani-Hackathon-kishore
```

Now you’ll see all the folders:
Defect Detection/, VehicleDetection/, bee-dynamics-3d/, etc.


---

##  Getting Started

### Prerequisites

- Python 3.8+
- pip packages:
 
```
pip install numpy matplotlib pandas seaborn torch torchvision ultralytics gitpython
```
## Project 1: Defect Detection
Folder: Defect Detection/

How to Run
```
cd "Defect Detection"
python detection.py
```
What it does: Reads provided images and detects defects visually or using a rule-based logic.

Results are displayed inline or saved based on detection.py logic.

Read more in Defect Detection/README.md.

## Project 2: Vehicle Detection & Classification

https://www.kaggle.com/code/lucario73/p3-done

Folder: VehicleDetection/

How to Run
Pre-process Data:

```
cd VehicleDetection
python preprocess.py
```
This script cleans up images and XML annotations, converting them to YOLO format and splitting into train/val/test.

Visualize Dataset:

```
python visualize_data.py
```
Generates distribution plots such as class counts and bounding box statistics.

Train YOLOv5 Model:
```
python train.py
```
Launches YOLOv5 training (ensure YOLOv5 repo and weights are prepared as referenced in script).

Evaluate Results:

```
python evaluate.py
```
Produces loss curves, mAP, precision/recall, and example inference visuals.

Alternatively, open and run Vehicle Detection notebook.ipynb in Kaggle or Jupyter for a guided, interactive workflow.

For more details, see VehicleDetection/README.md.

## Project 3: Bee Dynamics in 3D (Lorenz Attractor)
Folder: bee-dynamics-3d/

How to Run
```
cd bee-dynamics-3d
python LorenzAttractor3D.py
```
This script simulates 3D chaotic paths (butterfly/bee movement) using the Lorenz system and plots the result.

The result.png image shows a sample trajectory.

For background and details, check bee-dynamics-3d/README.md.

Summary
Project	Folder	Run Script
Defect Detection	Defect Detection/	python detection.py
Vehicle Detection	VehicleDetection/	preprocess.py, train.py, evaluate.py, or use the notebook
Bee Dynamics (Lorenz 3D)	bee-dynamics-3d/	python LorenzAttractor3D.py

License & Contributions
All projects are organized with modular design principles and include detailed documentation. Feel free to open issues or submit pull requests — feedback is most welcome!

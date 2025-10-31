# 🔥Fire Smoke Segmentation💨

An AI-driven solution for **early fire detection and monitoring**, leveraging state-of-the-art **YOLO segmentation models** to automatically detect and segment fire and smoke in images and videos.

This project aims to support **safety monitoring**, reduce **response times**, and improve **environmental protection** through real-time computer vision.

## Features
* **Training Pipeline** → Train the detection model directly within the app
* **Upload Images** → Detect and segment fire/smoke regions
* **YOLO-Seg Powered** → Efficient, lightweight, and accurate segmentation
* **Streamlit Interface** → Easy-to-use web application with interactive results
* **Annotated Outputs** → Visualization with masks and labels

## Project Setup
### Prerequisites
- Python 3.10+
- PyTorch Cuda 1.8+ [[Download PyTorch Cuda](https://pytorch.org/)]
- Compatible cuda toolkit and cudnn installed on your machine [[Nvidia GPU Capability](https://developer.nvidia.com/cuda-gpus)] [[Download Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit)] (Note: You must have a [Nvidia Developer Account](https://developer.nvidia.com/login))
- Anaconda or Miniconda installed on your machine [[Download Anaconda](https://www.anaconda.com/download)]

### Installation
1. **Clone the repository:**
```bash
git clone https://github.com/Kanon14/fire-smoke-segmentation.git
cd fire-smoke-segmentation
```

2. **Create and activate an environment:**
```bash
# Conda environment
conda create -n fire_smoke python=3.10 -y
conda activate fire_smoke

# uv dependencies
uv venv --python 3.10
.venv/Scripts/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Workflow
The project workflow is designed to facilitate a seamless transition from development to deployment:
1. `constants`: Manage all fixed variables and paths used across the project.
2. `entity`: Define the data structures for handling inputs and outputs within the system.
3. `components`: Include all modular parts of the project such as data preprocessing, model training, and inference modules.
4. `pipelines`: Organize the sequence of operations from data ingestion to the final predictions.
5. `application`: This is the main executable script that ties all other components together and runs the whole pipeline.

## Workflow
The project workflow is designed to facilitate a seamless transition from development to deployment:
1. `constants`: Manage all fixed variables and paths used across the project.
2. `entity`: Define the data structures for handling inputs and outputs within the system.
3. `components`: Include all modular parts of the project such as data preprocessing, model training, and inference modules.
4. `pipelines`: Organize the sequence of operations from data ingestion to the final predictions.
5. `application`: This is the main executable script that ties all other components together and runs the whole pipeline.


## How to Run
1. **Execute the project:**
```bash
streamlit run streamlit_app.py
```
2. **Then, access the application via your web browser:**
```bash
open http://localhost:<port>
```

## Acknowledgements
- **[Roboflow](https://roboflow.com/):** For dataset hosting and augmentation tools.
- **[Ultralytics](https://www.ultralytics.com/):** For the YOLO object detection framework.
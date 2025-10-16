# YOLOv8 Object Detection for Autonomous Driving

This project fine-tunes a pre-trained YOLOv8n model on the KITTI dataset to perform object detection for autonomous driving scenarios. The goal is to accurately detect and classify objects commonly found in road scenes, such as cars, pedestrians, and cyclists.

<img width="1508" height="928" alt="image" src="https://github.com/user-attachments/assets/a30c4186-726f-461e-a0e3-59dbcb22ce8a" />


## Features

-   **State-of-the-Art Model**: Utilizes YOLOv8, a powerful and efficient object detection model.
-   **Autonomous Driving Dataset**: Trained and validated on the KITTI dataset, a standard benchmark for computer vision in autonomous driving.
-   **End-to-End Workflow**: Demonstrates the complete process from data preparation and configuration to model training, validation, and prediction.
-   **Performance Evaluation**: Generates and visualizes key metrics, including a confusion matrix and validation results.

## Technologies Used

-   **Python 3.x**
-   **Ultralytics YOLOv8**: The core framework for object detection.
-   **Kaggle Hub**: For easy and programmatic dataset access.
-   **Scikit-learn**: For splitting the dataset into training and validation sets.
-   **PyTorch**: The backend deep learning framework for YOLOv8.
-   **Pandas & NumPy**: For data handling and manipulation.
-   **Matplotlib & Seaborn**: For plotting and visualizing results.

## Project Workflow

1.  **Data Preparation**: The KITTI dataset (in YOLO format) is downloaded from Kaggle Hub.
2.  **Dataset Splitting**: The images and corresponding labels are split into training (90%) and validation (10%) sets.
3.  **YAML Configuration**: A `kitti.yaml` file is programmatically generated to define the dataset paths, number of classes, and class names for the YOLOv8 trainer.
4.  **Model Training**: A pre-trained `yolov8n.pt` model is loaded and fine-tuned on the custom KITTI training set for 10 epochs.
5.  **Evaluation & Prediction**: The trained model is validated on the test set, and its performance is visualized. Finally, predictions are run on random sample images from the validation set to demonstrate its capabilities.

## Setup and Installation

Follow these steps to set up the project locally.

### 1. Clone the repository:
```bash
git clone [https://github.com/Mannan-15/YOLOv8-Object-Detection-for-Autonomus-Driving.git](https://github.com/Mannan-15/YOLOv8-Object-Detection-for-Autonomus-Driving.git)
cd YOLOv8-Object-Detection-for-Autonomus-Driving
```

### 2. Configure Kaggle API (Required for Dataset Download):
This project uses `kagglehub` to download the dataset automatically. You need to have your Kaggle API credentials set up.
-   Go to your Kaggle account settings, find the "API" section, and click "Create New Token". This will download a `kaggle.json` file.
-   Place this `kaggle.json` file in the appropriate directory (e.g., `~/.kaggle/` on Linux/macOS or `C:\Users\<Your-Username>\.kaggle\` on Windows).

### 3. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install the required libraries:
```bash
pip install -r requirements.txt
```

## ▶️ Usage

To run the project, execute the Python script or Jupyter Notebook. The script will handle everything from downloading the data to training the model and saving the prediction results.
```bash
python your_script_name.py
# or open and run the .ipynb file in a Jupyter environment.
```
The training results, including performance graphs and sample prediction images, will be saved in a new directory named `yolov8n-kitti`.

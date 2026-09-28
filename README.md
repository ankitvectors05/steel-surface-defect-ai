# AI-Based Steel Surface Defect Detection & Segmentation

An AI-based computer vision system for detecting and segmenting surface defects in steel images using YOLO segmentation, OpenCV, and Streamlit.

## Project Overview

Surface defects in steel can affect product quality and require reliable inspection.

This project develops a prototype automated inspection system that:

- Detects surface defects in steel images
- Classifies defects into four defect classes
- Localizes defects using segmentation masks
- Calculates the percentage of image area affected by defects
- Provides confidence scores for predictions
- Provides an interactive Streamlit interface for inspection

## Pipeline

```text
Steel Surface Image
        ↓
Image Input
        ↓
YOLO Segmentation Model
        ↓
Defect Detection + Classification
        ↓
Segmentation Mask
        ↓
Defect Area Calculation
        ↓
Inspection Results
        ↓
Streamlit Application
```

## Dataset

The project uses the **Severstal Steel Defect Detection** dataset from Kaggle.

Dataset:  
https://www.kaggle.com/c/severstal-steel-defect-detection/data

The original training dataset contains 12,568 steel surface images and four defect classes.

For this project, 8,000 images were selected:

- Training: 6,400 images
- Validation: 1,600 images

The original RLE-encoded defect annotations were converted into segmentation annotations suitable for YOLO training.

## Model

**YOLO26n-seg**

The model was trained using Google Colab with a Tesla T4 GPU.

### Training Configuration

- Training images: 6,400
- Validation images: 1,600
- Epochs: 10
- Image size: 640
- Task: Instance segmentation

## Validation Results

The model achieved the following validation results after 10 epochs:

| Metric | Result |
|---|---:|
| Box Precision | 48.5% |
| Box Recall | 31.8% |
| Box mAP50 | 33.0% |
| Box mAP50-95 | 15.1% |
| Mask Precision | 57.3% |
| Mask Recall | 30.7% |
| Mask mAP50 | 29.9% |
| Mask mAP50-95 | 11.2% |

These results represent a baseline prototype rather than a production-ready inspection system.

## Application Features

The Streamlit application allows a user to upload a steel surface image and obtain:

- Number of detected defects
- Defect class
- Prediction confidence
- Defect area percentage
- Defect localization overlay
- Segmentation mask

## Project Structure

```text
steel-surface-defect-ai/
│
├── models/
│   └── best.pt
│
├── notebooks/
│   └── steel_defect_yolo_segmentation.ipynb
│
├── screenshots/
│
├── app.py
├── inference.py
├── preprocessing.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd steel-surface-defect-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload a steel surface image and click **Inspect Steel Surface** to run the inspection.

## Application Workflow

```text
Upload Steel Image
        ↓
Image Processing
        ↓
YOLO Segmentation
        ↓
Defect Detection
        ↓
Defect Classification
        ↓
Segmentation Mask
        ↓
Defect Area Calculation
        ↓
Inspection Results
```

## Output

For an uploaded steel surface image, the application provides:

- Detected defect count
- Defect class
- Prediction confidence
- Defect area percentage
- Defect localization overlay
- Predicted segmentation mask

## Limitations

The current implementation is a baseline prototype and has several limitations:

- The model was trained for only 10 epochs.
- The dataset contains significant class imbalance.
- Recall is relatively low for some defect classes.
- The current project does not use an independent test set with publicly available ground-truth labels.
- The model has not been calibrated for an industrial production environment.
- No production PASS/FAIL threshold has been defined because such a threshold would require an appropriate engineering quality specification.

## Future Improvements

Potential improvements include:

- Longer model training
- Class-aware sampling
- Improved data augmentation
- Hyperparameter tuning
- More rigorous train/validation/test splitting
- Comparison with larger YOLO segmentation models
- Improved detection of underrepresented defect classes
- Model optimization for deployment
- Industrial quality-threshold calibration
- Evaluation on additional real-world steel inspection images

## Technologies Used

- Python
- YOLO26n-seg
- Ultralytics
- PyTorch
- OpenCV
- NumPy
- Streamlit
- Google Colab

## Training Environment

Model training was performed using Google Colab with GPU acceleration.

```text
GPU: Tesla T4
Training Images: 6,400
Validation Images: 1,600
Epochs: 10
Image Size: 640
Task: Instance Segmentation
```

## Local Inference

The trained model is stored locally at:

```text
models/best.pt
```

The Streamlit application uses this trained model to perform inference on newly uploaded steel surface images.

## Project Status

**Completed Prototype**

The current version demonstrates an end-to-end workflow from steel surface image input to defect detection, segmentation, defect-area calculation, and interactive visualization.

## Author

**Ankit Kumar**

B.Tech Mechanical Engineering  
NIT Jamshedpur
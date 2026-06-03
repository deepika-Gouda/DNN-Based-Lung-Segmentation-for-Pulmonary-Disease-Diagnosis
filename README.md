# DNN-Based Lung Segmentation for Pulmonary Disease Diagnosis

## Overview

This project is a Deep Neural Network (DNN)-based web application for automated lung segmentation and pulmonary disease diagnosis from chest X-ray images. The system combines image segmentation and disease classification techniques to assist in the early detection and analysis of lung-related diseases such as COVID-19, Pneumonia, and Lung Opacity.

The application provides an interactive web interface where users can upload chest X-ray images and receive segmentation results, disease predictions, visual explanations, and diagnostic reports.

---

## Features

* Lung region segmentation from chest X-ray images
* Pulmonary disease classification
* Detection of:

  * COVID-19
  * Pneumonia
  * Lung Opacity
  * Normal cases
* Web-based user interface using Flask
* User authentication and dashboard management
* Grad-CAM visualization for model explainability
* Segmentation overlay generation
* Diagnostic report generation in PDF format
* Role-based access (Admin, Doctor, User)

---

## Technologies Used

### Backend

* Python
* Flask

### Deep Learning

* PyTorch
* TensorFlow / Keras
* OpenCV
* NumPy

### Frontend

* HTML
* CSS
* JavaScript
* Bootstrap

### Database

* SQLite

---

## Project Structure

```text
├── app.py
├── models.py
├── losses.py
├── project_model/
│   ├── predict.py
│   ├── resnet_seg.py
│   └── __init__.py
├── templates/
├── static/
│   ├── css/
│   ├── js/
├── scripts/
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/deepika-Gouda/DNN-Based-Lung-Segmentation-for-Pulmonary-Disease-Diagnosis.git
cd DNN-Based-Lung-Segmentation-for-Pulmonary-Disease-Diagnosis
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## Workflow

1. Upload a chest X-ray image.
2. Perform lung segmentation.
3. Generate segmentation overlays.
4. Predict disease category.
5. Visualize model attention using Grad-CAM.
6. Generate diagnostic report.

---

## Applications

* Clinical decision support
* Medical image analysis
* Pulmonary disease screening
* Academic research in medical AI
* Healthcare diagnostics

---

## Future Enhancements

* Multi-disease classification
* Real-time cloud deployment
* Integration with hospital management systems
* Improved segmentation accuracy
* Explainable AI enhancements
* Mobile application support



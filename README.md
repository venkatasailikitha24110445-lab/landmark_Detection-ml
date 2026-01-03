Landmark Detection using Deep Learning
Developer: Likitha Innamuri
Project Overview

This repository implements an end-to-end landmark image classification system using deep learning and transfer learning.
The model is trained to recognize landmarks from images and the project includes:

data loading and preprocessing

model training

model evaluation

confusion-matrix generation

REST API inference service

Docker deployment support

ONNX model export

basic unit-test scaffold

This project is organized to reflect industry-style ML engineering practices.

Features

Transfer learning using MobileNetV2

Training and validation pipeline

Model saving and loading

Evaluation on test dataset

Confusion matrix visualization

Single-image prediction script

FastAPI-based REST inference API

Dockerfile for containerized deployment

Script to export model to ONNX

Configuration files for training settings

Project Structure
landmark-detection-ml-elite
│
├── scripts/                 # training, evaluation, prediction, ONNX export
├── api/                     # FastAPI application for inference
├── models/                  # trained models (add after training)
├── data/                    # dataset (not included)
├── results/                 # saved plots and evaluation outputs
├── tests/                   # unit test scaffold
├── configs/                 # training configuration files
├── requirements.txt
├── Dockerfile
└── README.md

Requirements Installation

Install dependencies:

pip install -r requirements.txt


Recommended environment:

Python 3.8+

GPU supported machine (optional but helpful)

Dataset Format

Expected CSV format:

| id (image filename) | landmark_id (label) |

Images are expected in directories such as:

data/train_images/
data/test_images/

Train the Model

Run:

python scripts/train.py


This will:

train the landmark classification model

save trained model to models/landmark_model.h5

generate learning-curve plots in results/

Evaluate the Model

Run:

python scripts/evaluate.py


This will:

load the saved model

evaluate on test data

generate confusion matrix in results/

Predict Single Image

Run:

python scripts/predict_single_image.py --img your_image.jpg


Outputs predicted class in the terminal.

FastAPI Inference Service

Start API server:

uvicorn api.app:app --reload


Interactive API docs available at:

http://localhost:8000/docs

Docker Deployment

Build image:

docker build -t landmark-api .


Run container:

docker run -p 8000:8000 landmark-api

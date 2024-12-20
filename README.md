# Deepfake Detection Using Binary Neural Networks (BNNs)

This project implements a deepfake detection model using the [BNext](https://arxiv.org/pdf/2211.12933) architecture on the [CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) dataset. The goal is to classify images as real or AI-generated synthetic images.

## Setup

### Step 1: Clone the repository

Run this to clone the repository

```bash
git clone https://github.com/MYAzrak/CV-Project-G5.git
```

### Step 2: Install Dependencies

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### Step 3: Download the CIFAKE dataset

Download the [CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) dataset by cloning their repository.

```bash
git clone https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images
```

### Step 4: Download best_model.pth

Download the best pre-trained model to use it in testing from 'Pre-trained model' release.

### Step 5: Modify the paths

Modify train_dir, test_dir, and model_path in config.py to point to your specific dataset and pre-trained_model paths.

### Step 6: Try the model

Run train.py, test.py, or inference.py.

## Collaborators

This project was worked on with the contributions of [arcarum](https://github.com/arcarum), [Adithya](https://github.com/AdiKk69), and [Abdullah Shahid](https://github.com/Abdullah-Shahid01).

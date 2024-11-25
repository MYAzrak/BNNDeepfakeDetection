# Deepfake Detection Using Binary Neural Networks (BNNs)

This project implements a deepfake detection model using the [BNext](https://arxiv.org/pdf/2211.12933) architecture on the [CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) dataset. The goal is to classify images as real or AI-generated synthetic images.

---

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

Download the CIFAKE dataset by cloning their repository

```bash
git clone https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images
```

### Step 4: Download best_model.pth

[Download](https://drive.google.com/file/d/1y7iRO4dc7VhysOMyVt4hv-9A7obbwVCp/view?usp=sharing) the best pre-trained model to use it in testing

### Step 5: Modify the datasets' paths

Modify train_dir and test_dir in config.py to point to your specific dataset paths

### Step 5: Try the model

Run train.py, test.py, or inference.py

# Skin Lesion Classification

An end-to-end machine learning pipeline for classifying dermoscopic images of skin lesions into 8 diagnostic categories. The project leverages a dual-stage approach: **deep learning-based automated segmentation** via a U-Net architecture, followed by **clinically-grounded ABCD feature extraction** and an **SVM classifier**.

---

## Overview

Early detection of skin cancer dramatically improves patient outcomes. This project implements an automated, feature-engineered classification pipeline built on top of the International Skin Imaging Collaboration (ISIC) dataset. 

The pipeline classifies lesions into **8 diagnostic categories**:
1. Melanoma
2. Melanocytic nevus
3. Basal cell carcinoma
4. Actinic keratosis
5. Benign keratosis
6. Dermatofibroma
7. Vascular lesion
8. Squamous cell carcinoma

---

## Pipeline Architecture

The entire pipeline is consolidated into a single, cohesive Jupyter notebook: `skin_lesion_pipeline.ipynb`. It executes three core phases:


### 1. Segmentation (Predict missing masks)
Not all images in the dataset include ground-truth masks. 
* **Model:** A U-Net architecture (encoder-decoder with skip connections).
* **Training:** Validated on a subset with ground-truth masks using a combined **Binary Cross-Entropy + Dice Loss** with early stopping.
* **Inference:** Automatically generates and processes missing segmentation masks for the remaining dataset.

### 2. Feature Extraction
Features are extracted dynamically from the combined image-mask pairs using the clinical **ABCD rule**:
* **Asymmetry:** Shape and color asymmetry computed across vertical and horizontal axes.
* **Border:** ANOVA F-test evaluating pixel intensity variations along the lesion boundary.
* **Color:** Mean, standard deviation, and skewness of RGB channels within the masked region.
* **Dimension:** Bounding box metrics, total lesion area, and equivalent diameter.

**Additional Features:**
* **Texture:** Local Binary Patterns (LBP) at 3 discrete scales alongside Gray-Level Co-occurrence Matrix (GLCM) metrics (contrast, dissimilarity, homogeneity, energy, entropy).
* **Frequency Domain:** Fourier Transform magnitude spectrum metrics (mean and standard deviation).
* **Metadata:** Patient demographics including age, sex, and anatomical site position.

### 3. Classification
* **Model:** Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel.
* **Imbalance Handling:** Managed via `class_weight='balanced'` inside the SVM to counter heavy class imbalances.
* **Evaluation:** Assessed using Macro F1-score, class-specific recall, and Weighted Categorization Accuracy.

---

## Results

Performance evaluated on an 80/20 stratified validation split:

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 0.66 |
| **Weighted F1-Score** | 0.68 |

---

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/skin-lesion-classification.git](https://github.com/your-username/skin-lesion-classification.git)
   cd skin-lesion-classification
Install Dependencies:

Bash
pip install -r requirements.txt
Data Structure
Download the dataset components from the ISIC Archive and arrange them as follows:

Plaintext
skin-lesion-classification/
├── Train/
│   └── Train/
│       ├── ISIC_XXXXXXX.jpg
│       └── ISIC_XXXXXXX_seg.png    # (Where ground truth is available)
├── Test/
│   └── Test/
│       └── ISIC_XXXXXXX.jpg
├── metadataTrain.csv
├── metadataTest.csv
├── SampleSubmission.csv
└── gpu_unet_improved.h5            # Pre-trained U-Net weights
Usage
Launch Jupyter Notebook and open the master pipeline script:

Bash
jupyter notebook skin_lesion_pipeline.ipynb
Execute the cells sequentially. If you are utilizing the pre-trained weights (gpu_unet_improved.h5), you can flag the model training cells to False to jump straight to automated mask prediction and feature processing.

Project Structure
Plaintext
skin-lesion-classification/
├── skin_lesion_pipeline.ipynb   # Master notebook containing the full pipeline
├── requirements.txt             # Project dependencies 
└── README.md                    # Project documentation

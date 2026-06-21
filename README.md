# skin-lesion-UNet-classification
Skin Lesion Classification
A machine learning pipeline for classifying dermoscopic images of skin lesions into 8 diagnostic categories, using clinically-grounded ABCD feature extraction and an SVM classifier.
Overview
Skin cancer is one of the most common cancers worldwide, and early detection dramatically improves outcomes. This project builds an automated classification pipeline on top of the ISIC dataset, a large public collection of dermoscopic images with clinical labels.
8 diagnostic classes:
#	Class
1	Melanoma
2	Melanocytic nevus
3	Basal cell carcinoma
4	Actinic keratosis
5	Benign keratosis
6	Dermatofibroma
7	Vascular lesion
8	Squamous cell carcinoma
---
Pipeline
The project is split across 3 notebooks that run in sequence:
```
1\_segmentation.ipynb       →    2\_feature\_extraction.ipynb    →    3\_classification.ipynb
Train UNet on available         Extract ABCD + texture              Train SVM, evaluate,
masks → predict missing         features from images                generate predictions
masks for all images            and their masks
```
1. Segmentation (`1\_segmentation.ipynb`)
Not all images in the dataset come with ground truth segmentation masks. A UNet (encoder-decoder with skip connections) is trained on the subset that does, then used to predict masks for the rest.
Loss: combined Binary Cross-Entropy + Dice Loss
Trained with early stopping on validation Dice coefficient
Predicted masks are saved to disk and used in the next step
2. Feature Extraction (`2\_feature\_extraction.ipynb`)
Features are extracted following the clinical ABCD rule:
Asymmetry — shape and color asymmetry computed by comparing halves of the lesion across both axes
Border — ANOVA F-test on pixel intensity differences along the lesion boundary
Color — mean, std, and skewness of RGB channels inside the mask
Dimension — bounding box dimensions, lesion area, equivalent diameter
Additional texture features:
LBP (Local Binary Pattern) at 3 scales
GLCM (Gray-Level Co-occurrence Matrix) — contrast, dissimilarity, homogeneity, energy, entropy
Fourier Transform — mean and std of the magnitude spectrum
Demographic metadata (age, sex, anatomical position) is also included as features.
3. Classification (`3\_classification.ipynb`)
An RBF-kernel SVM is trained on the scaled feature matrix.
`class\_weight='balanced'` handles the significant class imbalance
Evaluation uses both standard metrics (F1, recall per class) and the challenge's Weighted Categorization Accuracy, which up-weights minority classes
---
Results
Validation set performance (80/20 stratified split):
Metric	Score
Weighted Accuracy	—
Overall Accuracy	0.66
Weighted F1	0.68
Macro F1	0.53
Per-class breakdown:
Class	Precision	Recall	F1
Melanoma	0.56	0.58	0.57
Melanocytic nevus	0.90	0.75	0.81
Basal cell carcinoma	0.61	0.60	0.61
Actinic keratosis	0.30	0.59	0.40
Benign keratosis	0.46	0.48	0.47
Dermatofibroma	0.21	0.69	0.32
Vascular lesion	0.74	0.66	0.69
Squamous cell carcinoma	0.26	0.51	0.34
The model performs best on Melanocytic nevus (the majority class) and Vascular lesion, and struggles most with minority classes like Dermatofibroma and Squamous cell carcinoma — expected given the heavy class imbalance.
---
Setup
```bash
git clone https://github.com/your-username/skin-lesion-classification.git
cd skin-lesion-classification

pip install -r requirements.txt
```
Data
Download the ISIC dataset and place it in the following structure:
```
skin-lesion-classification/
├── Train/
│   └── Train/
│       ├── ISIC\_XXXXXXX.jpg
│       └── ISIC\_XXXXXXX\_seg.png   ← (where available)
├── Test/
│   └── Test/
│       └── ISIC\_XXXXXXX.jpg
├── metadataTrain.csv
├── metadataTest.csv
└── SampleSubmission.csv
```
Pre-trained UNet weights
The UNet was trained on a GPU. If you want to skip retraining, download the pre-trained weights:
> \*\*\[Download `unet\_best\_model.keras`]\*\* ← \*(add your Google Drive / release link here)\*
Place the file in the root of the project directory, then run only the mask prediction cells in `1\_segmentation.ipynb`.
---
Run
Run the notebooks in order:
```bash
jupyter notebook
```
`1\_segmentation.ipynb`
`2\_feature\_extraction.ipynb`
`3\_classification.ipynb`
Each notebook saves its output as CSV files for the next step to pick up.
---
Project Structure
```
skin-lesion-classification/
├── 1\_segmentation.ipynb          # UNet training and mask prediction
├── 2\_feature\_extraction.ipynb    # ABCD + texture feature extraction
├── 3\_classification.ipynb        # SVM training, evaluation, predictions
├── requirements.txt
└── README.md
```
---
References
Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)
Esteva et al., Dermatologist-level classification of skin cancer with deep neural networks, Nature (2017)
ISIC Archive: https://www.isic-archive.com/
Celebi et al., Dermoscopy image analysis overview

# Skin Lesion Classification & Segmentation
### Final Project Submission for COSC 6370 - MEDICAL IMAGING

---
## Overview
-	**HAM10000_training.ipynb:**
This file consists of all the python code and functions required to create the UNET segmentation model, train, test it and perform feature extraction. It also contains the code for training a SVM classifier for skin lesion classification.
-	**Project_GUI.py:**
This python file consists of the code required for the GUI implementation and involves invoking the saved model files for the UNET and SVM algorithms. 
-	**model-skin-lesion-segmentation-org2000.h5:**
Represents the saved model file for the trained UNET algorithm with 2000 data points
-	**svm_model_poly.pkl:**
Represents the saved model file for the trained SVM polynomial classifier for the classification of skin lesion images.
-	**Data folder:**
Consists of the link to original HAM10000 dataset, metadata file and the empty train, test folders that will be populated with images when code is executed using HAM10000_training.ipynb file.

## Usage
The code in this repository can be used by following the below steps:

STEP 1: Download data from the link "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T" and click on the button "Access dataset" to the top right corner.
Select Original Format zip from the list dropdown.

STEP 2: Unzip the contents of the downloaded zip file and extract the contents of HAM10000_images_part_1.zip and HAM10000_images_part_2.zip into the data/HAM10000 folder.

STEP 3: Next unzip the contents of the file HAM10000_segmentations_lesion_tschandl.zip into the folder HAM10000_segmentations_lesion_tschandl located in the main project directory.

STEP 4: Run the HAM10000_training.ipynb notebook to train and save the UNET and SVM model files.

STEP 5: Install the required libraries from the requirements.txt file using "pip -r requirements.txt"

STEP 6: Run the file Project_GUI.py in the same folder to view the GUI. The Images required for testing in the GUI can be selected from data/test folder. The output images are saved to folder data/user_inputs

## Quick Testing of Project
The project can also be run directly without model training by invoking the **Project_GUI.py** file post installing the required libraries from requirement.txt file.

Run the .py file and select an image from the data/test folder to perform all the pre-processing, segmentation and classification tasks on the chosen skin lesion image.
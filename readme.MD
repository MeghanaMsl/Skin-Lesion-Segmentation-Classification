# Skin Lesion Classification & Segmentation
---
## Overview
In this project, a Convolutional Neural Network (CNN’s) for accurate skin lesion segmentation using U-net algorithm is created. 
Further the ROI is processed to collect a series of features, scanning the skin area of interest for asymmetry (A), border irregularity (B), colors variegation (C), diameter (D) and texture (T) using efficient ABCDT feature extraction technique. 
The extracted features are then fed into the Support Vector Machine (SVM) classifier for classification

## File Description
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

STEP 6: Run the file Project_GUI.py in the same folder to view the GUI. The Images required for testing in the GUI can be selected from data/test folder. The output images are saved to folder data/gui_outputs

## Quick Testing of Project
The project can also be run directly without model training by invoking the **Project_GUI.py** file post installing the required libraries from requirement.txt file.

Run the .py file and select an image from the data/test folder to perform all the pre-processing, segmentation and classification tasks on the chosen skin lesion image.

The output files are automatically saved to the data/gui_outputs folder.

## Result
The segmentation masks obtained by our trained UNet model compared against the ground truth masks from the HAM10000 dataset are as follows:
![result_segmentation.png](result_segmentation.png?raw=true "Results")

The screenshot below shows the PyQT GUI with all the sequence of steps that have been carried out in this project. 
![result.png](result.png?raw=true "Results")

- The first image is the original skin lesion image
- The second image represents the grayscale converted image.
- The third image shows the result of performing the morphology operation on the grayscale image. 
- The fourth image represents the inpainted image with the thresholds calculated from the morphology operation. At this step any noise due to hair is eliminated.
- The fifth image represents the Gaussian blurred image which is sent as an input to our UNET model. 
- The final image represents the segmentation map predicted by our UNET model.
- Lastly, clicking the “View Classification” button would show the results of the original image category.
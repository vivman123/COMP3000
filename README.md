# COMP3000

# Vivek Raval
# Student ID: 10774366
# Galaxy Morphological Classifier

# Project Vision:
A deep learning pipeline that can accurately classify galaxy morphologies (Spiral, Elliptical, Irregular) based on the Galaxy Zoo 2 dataset. The project will fill the gap between training and practical applications through a Flask web app that integrates visual explanation techniques (Grad-CAM) to deliver clear and accurate astronomical predictions.

# Repository Structure:
The folder structure for COMP3000 Computing Project.
1. 'COMP3001.py' - PyTorch script for training the model. Data from the Galaxy Zoo 2 dataset is used for model training with a balanced dataset, augmented data, and dropout layer for regularisation.
2. 'app.py' - Flask web application that hosts the trained model. The application provides an inference pipeline with standardised image resizing based on the GZ2 424px to 200px crop strategy. The application also employs PyTorch Grad-CAM for generating visual heatmaps of the models’ predictions. This is also includes the HTML file 'index.html' for the handling of the file uploads.
   
# Libraries and open source tools used: 
python 3.13
torch
torchvision
pandas
scikit-learn
tqdm
Pillow
numpy
flask
matplotlib
opencv-python
grad-cam

# Setup
To run the Webapp Create a venv enviroment, install the requirements.txt file requirements, then run the app.py file ensuring there is a pth file for the web app to use. 
To run the training model follow the same steps however run the COMP3000.py file and ensure you've downloaded the GZ2 Dataset from Kaggle (https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images) including the csv's for filename mapping etc

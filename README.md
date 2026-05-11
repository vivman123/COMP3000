# Galaxy Morphological Classifier

**Course:** COMP3000 Project  
**Author:** Vivek Raval  
**Student ID:** 10774366  

---

## Project Vision

A deep learning pipeline designed to accurately classify galaxy morphologies (Spiral, Elliptical, Irregular) using the **Galaxy Zoo 2 (GZ2)** dataset. This project bridges the gap between model training and practical application through a Flask web application. It integrates visual explanation techniques, specifically **Grad-CAM**, to deliver clear, interpretable, and accurate astronomical predictions.

## Repository Structure

* `COMP3000.py`  
  PyTorch script for training the model. It utilizes data from the Galaxy Zoo 2 dataset for model training, featuring a balanced dataset, data augmentation, and dropout layers for regularization.
* `app.py`  
  The Flask web application that hosts the trained model. It also employs PyTorch Grad-CAM to generate visual heatmaps of the model's predictions.
* `templates/index.html`  
  The front-end HTML interface handling user file uploads for the web application.

## Tech Stack & Libraries

* **Language:** Python 3.13
* **Machine Learning:** PyTorch (`torch`, `torchvision`), Scikit-Learn
* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, OpenCV (`opencv-python`), Pillow, Grad-CAM
* **Web Framework:** Flask
* **Utilities:** tqdm

## Setup & Installation

### 1. Environment Setup
First, clone the repository and set up a virtual environment:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```
### 2. Running the Web Application
Ensure you have a trained model file (.pth) available in the root directory for the web app to use.

```bash
# Start the Local Flask server
python app.py
```
### 3. Training the Model
If you want to train the model from scratch, you must first download the dataset.

Download the GZ2 Dataset from Kaggle: [Galaxy Zoo 2 Images](https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images).

Ensure you extract the images and the CSV files (for filename mapping) into the correct data directory as specified in the script.

Run the training script:

```bash
python COMP3000.py
```

# Plant Disease Predictor

This repository contains a Plant Disease Predictor, a web application that uses a deep learning model to predict diseases in plants based on input images. The project is built with Python and Streamlit, and leverages a pre-trained model stored in the `plant_disease_model.h5` file.

## Project Structure

- **`Plant_Disease_Model.ipynb`**: Jupyter notebook used to train the deep learning model and generate the `plant_disease_model.h5` file.
- **`app.py`**: Python script that runs the web application.
- **`plant_disease_model.h5`**: Pre-trained model for plant disease prediction.

## Dataset

The dataset used to train the model is available on Kaggle: [Plant Village Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease). You can download it and use it to retrain or experiment with the model.

## Getting Started

Follow the steps below to set up and run the project.

### Prerequisites

- Python 3.8 or later
- Required libraries:
  - Streamlit
  - TensorFlow/Keras
  - Numpy
  - OpenCV
  - Pillow

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Nishaad27/Plant-Disease-Detector.git
   cd Plant-Disease-Detector
   ```

2. Install the dependencies:
   ```bash
   pip install streamlit tensorflow numpy opencv-python pillow
   ```

### Running the Application

1. Ensure the `plant_disease_model.h5` file is present in the project directory. If not, generate it by running the `Plant_Disease_Model.ipynb` notebook.

2. Start the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL provided in the terminal (typically `http://localhost:8501/`).

4. Upload an image of a plant leaf, and the application will predict the disease (if any).

### Training the Model

To retrain the model or modify it:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Open the `Plant_Disease_Model.ipynb` notebook in Jupyter Notebook or JupyterLab.
3. Follow the steps in the notebook to train the model and save it as `plant_disease_model.h5`.

## Project Files

- **`plant_disease_model.h5`**: Pre-trained model file.
- **`app.py`**: Streamlit web application script.
- **`Plant_Disease_Model.ipynb`**: Notebook for training and saving the model.

## Technologies Used

- **Python**: Core programming language.
- **TensorFlow/Keras**: Deep learning framework for training the model.
- **Streamlit**: Web framework for the application.
- **OpenCV**: Image processing library.

## Repository

You can access the complete project on GitHub: [Plant Disease Detector](https://github.com/Nishaad27/Plant-Disease-Detector.git)

## Acknowledgements

- Dataset: [Plant Village Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Special thanks to the open-source community for the libraries and frameworks used in this project.


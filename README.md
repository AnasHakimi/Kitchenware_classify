# Kitchenware Classifier AI

A deep learning-based web application that classifies kitchenware items (Forks, Knives, Plates, Spoons) using a Convolutional Neural Network (CNN). The system is built with **PyTorch** for the backend model and **FastAPI** for the web server, featuring a premium, modern frontend.

## Features
- **Deep Learning Model**: Custom CNN architecture implemented in PyTorch, replicating a MATLAB reference model.
- **FastAPI Backend**: High-performance, asynchronous web server for handling predictions.
- **Premium Frontend**: Modern, responsive UI with dark mode, glassmorphism effects, and smooth animations.
- **Real-time Prediction**: Drag & drop interface for instant image classification.

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AnasHakimi/Kitchenware_classify.git
   cd Kitchenware_classify
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model (Optional)
The repository comes with a pre-trained model (`kitchenware_model.pth`). If you wish to retrain the model on your own dataset:

1. Ensure your dataset is organized in the `Dataset` folder (or update `train.py` with your data path).
2. Run the training script:
   ```bash
   python train.py
   ```
   This will save a new `kitchenware_model.pth` file.

### 2. Running the Application
Start the FastAPI server:
```bash
python -m uvicorn app:app --reload --port 8000
```

### 3. Using the Web Interface
1. Open your web browser and navigate to `http://localhost:8000`.
2. Drag and drop an image of a kitchenware item (fork, knife, plate, or spoon) into the upload area.
3. Click **Analyze Image** to see the prediction results.

## Project Structure
- `app.py`: FastAPI application entry point.
- `model.py`: PyTorch CNN model definition.
- `train.py`: Script for training the model.
- `templates/index.html`: Main HTML file for the frontend.
- `static/style.css`: Premium styling for the application.
- `static/script.js`: Frontend logic for file handling and API communication.
- `requirements.txt`: List of Python dependencies.

## License
This project is open-source and available under the MIT License.

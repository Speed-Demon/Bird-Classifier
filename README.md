# Bird Call Classification using Audio Features

## Data Source
https://xeno-canto.org/

## Overview
This project focuses on classifying bird species based on their audio calls using machine learning techniques. The pipeline involves:
- Extracting audio features from bird call recordings.
- Applying data augmentation techniques.
- Training classification models including SVM, XGBoost, and RandomForest.
- Saving the best-performing model for inference.

## Features Extracted
- **MFCC (Mel-frequency cepstral coefficients)** and their derivatives.
- **Chroma features**
- **Spectral features** (centroid, bandwidth, rolloff)
- **Zero Crossing Rate**
- **Root Mean Square Energy (RMS)**

## Data Preprocessing & Augmentation
- Noise addition
- Time stretching
- Pitch shifting

## Dependencies
Ensure you have the following libraries installed:
```bash
pip install numpy pandas librosa scikit-learn xgboost joblib
```

## Dataset
The dataset consists of bird call audio files stored in a directory structure:
```
datasets/
    ├── species_1/
    │   ├── audio1.wav
    │   ├── audio2.mp3
    ├── species_2/
    │   ├── audio1.wav
    │   ├── audio2.mp3
```

## Running the Project
### 1. Extract Features & Create Dataset
```python
python extract_features.py
```
This will process the dataset and save training & testing CSV files.

### 2. Train the Model
```python
python train_model.py
```
This script:
- Loads the dataset
- Scales features using `StandardScaler`
- Trains models (SVM, XGBoost, RandomForest)
- Selects the best-performing model
- Saves the model (`best_model_bigdata.pkl`)

### 3. Model Evaluation
The script prints:
- Accuracy scores
- Classification reports for each model

### 4. Inference
To predict bird species from an audio file, use the trained model:
```python
import librosa
import joblib
import numpy as np
from extract_features import extract_features

# Load model & scaler
model = joblib.load("best_model_bigdata.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

# Load audio & extract features
y, sr = librosa.load("test_audio.wav", sr=22050, duration=5.0)
features = extract_features(y=y, sr=sr)
features_scaled = scaler.transform([features])

# Predict species
predicted_label = model.predict(features_scaled)
print("Predicted Species:", encoder.inverse_transform(predicted_label))
```

## Results & Performance
- The best-performing model is saved for future inference.
- The script provides accuracy and classification reports to analyze model performance.

## Future Improvements
- Experiment with deep learning models (CNNs, RNNs).
- Incorporate more sophisticated augmentations.
- Optimize hyperparameters for better performance.

## Author
Developed by **Jay, Kshitij, Aryan**.

## License
This project is open-source and available for educational purposes.


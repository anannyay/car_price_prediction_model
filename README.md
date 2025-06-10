# Car Price Prediction Model

A robust machine learning model for accurate car price prediction using regression algorithms including Linear Regression, Random Forest, and Gradient Boosting with comprehensive feature engineering.

## 🚗 Project Overview

This project develops a comprehensive car price prediction system that leverages multiple machine learning algorithms to provide accurate price estimates. The model incorporates advanced feature engineering techniques and provides both a training pipeline and a web-based prediction interface.

## 🔧 Features

- **Multiple ML Algorithms**: Linear Regression, Random Forest, Gradient Boosting
- **Advanced Feature Engineering**: Automated feature selection and preprocessing
- **Web API**: Flask-based API for real-time predictions
- **Data Pipeline**: Complete data cleaning and preprocessing pipeline
- **Model Persistence**: Serialized models for quick deployment

## 📁 Project Structure
car_price_prediction_model/
├── data_clean_and_pipeline.py    # Data preprocessing and feature engineering
├── model_training.py             # Model training and evaluation
├── predict_pipeline.py           # Prediction pipeline for new data
├── api.py                        # Flask API for web interface
├── index.html                    # Web interface for predictions
├── cleaned_data.csv              # Processed dataset
├── train-data.csv               # Training dataset
├── test-data.csv                # Testing dataset
├── model.pkl                    # Trained model
├── preprocessor.pkl             # Data preprocessing pipeline
├── selector.pkl                 # Feature selection pipeline
└── requirements.txt             # Python dependencies

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation
1. Clone the repository:

git clone https://github.com/yourusername/car_price_prediction_model.git
cd car_price_prediction_model

2. Install required packages:
pip install -r requirements.txt
Usage

Training the Model
python model_training.py

Making Predictions
python predict_pipeline.py

Running the Web API
python api.py

Then open your browser and go to http://localhost:5000
📊 Model Performance
The model uses ensemble methods combining multiple algorithms to achieve optimal performance:

Linear Regression: Baseline model for interpretability
Random Forest: Handles non-linear relationships and feature interactions
Gradient Boosting: Provides high accuracy through iterative improvement

🔍 Data Features
The model considers various car attributes including:

Make and model information
Year of manufacture
Mileage and condition
Engine specifications
Market factors

🛠️ Technical Details
Data Processing

Automated data cleaning and validation
Feature scaling and normalization
Categorical encoding for non-numeric features
Feature selection based on importance scores

Model Pipeline

Cross-validation for robust performance evaluation
Hyperparameter tuning for optimal results
Model serialization for deployment

📈 API Endpoints

GET / - Web interface for predictions
POST /predict - JSON API for price predictions

🤝 Contributing

Fork the repository
Create a feature branch
Make your changes
Commit your changes
Push to the branch
Create a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🔮 Future Enhancements

 Add more sophisticated feature engineering
 Implement deep learning models
 Add real-time market data integration
 Develop mobile app interface
 Add model interpretability features

 Note: Make sure to update the dataset regularly and retrain the model periodically to maintain prediction accuracy.
 





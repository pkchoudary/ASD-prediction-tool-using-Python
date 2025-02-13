# ASD Prediction Tool Using Python

## Overview
This project aims to predict the likelihood of Autism Spectrum Disorder (ASD) in individuals based on key behavioral and demographic attributes. The model is built using machine learning techniques to analyze patterns in the dataset.

## Features
- Data preprocessing and feature engineering
- Implementation of multiple machine learning models
- Model evaluation and performance metrics
- Interactive interface for prediction (optional)

## Dataset
The dataset used includes behavioral screening data and demographic attributes such as:
- Age in months
- Gender
- Ethnicity
- Jaundice history
- Family ASD history
- Questionnaire responses (A1 - A10)
- Screening score
- Who completed the test

## Technologies Used
- Python
- Pandas & NumPy (Data Handling)
- Scikit-learn (Machine Learning)
- Matplotlib & Seaborn (Data Visualization)
- Flask (Web Application - optional)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/ASD-Prediction-Tool-Using-Python.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ASD-Prediction-Tool-Using-Python
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Run the Jupyter Notebook to train and evaluate models:
   ```sh
   jupyter notebook
   ```
2. If using a web application, start the Flask server:
   ```sh
   python app.py
   ```
3. Open the browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Model Performance
The following machine learning models were trained and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

Evaluation metrics such as accuracy, precision, recall, and F1-score were used to compare the model performances.

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for suggestions and improvements.

## License
This project is licensed under the MIT License.

## Contact
For questions or collaboration, create an issue in this repository.


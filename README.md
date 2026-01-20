# Women's Diabetes Predictor

## Live Demo
[Open the Streamlit App](https://womens-diabetes-predictor-mwjjuyruxugdsthrqrahic.streamlit.app/)

## Screenshot
![App Screenshot](screenshot.png)

## Project Description
This project implements a machine learning-based female diabetes prediction model using a structured clinical dataset derived from female patients. The goal is to estimate the likelihood of diabetes based on routinely collected medical and demographic indicators.

## Dataset Overview
The model is trained on the following features:

- `Pregnancies`: Number of times pregnant  
- `Glucose`: Plasma glucose concentration  
- `BloodPressure`: Diastolic blood pressure (mm Hg)  
- `SkinThickness`: Triceps skinfold thickness (mm)  
- `Insulin`: 2-hour serum insulin (µU/ml)  
- `BMI`: Body mass index (weight in kg / height in m²)  
- `Pedigree`: Diabetes pedigree function (genetic risk indicator)  
- `Age`: Age in years  
- `Class`: Target variable indicating diabetes status `[1 (diabetic) / 0 (non-diabetic)]`  

## Model Description
The predictor uses supervised learning to classify whether a female patient is likely to have diabetes based on the input features.  
Data preprocessing includes handling missing or zero-valued clinical measurements, feature scaling, and exploratory analysis to identify key risk factors.  
Multiple classification algorithms were evaluated, and the final Bagging model was selected based on metrics such as accuracy, precision, recall, and ROC-AUC.

## Use Case
This model is intended for educational and analytical purposes, demonstrating how clinical data can be leveraged to build predictive health models.  
It can support early risk screening and highlight influential health indicators but **is not a diagnostic tool** and should not replace professional medical judgment.

## Outcome
The final model provides a binary prediction of diabetes risk along with the probability, enabling users to explore how physiological and lifestyle factors contribute to diabetes prevalence among women.

## Requirements
- Python 3.8+
- streamlit
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

## How to Run Locally
# Clone the repository
git clone https://github.com/MaryOluwatogbekele/Womens-Diabetes-Predictor.git
cd Womens-Diabetes-Predictor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

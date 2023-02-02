# Diabetes Predictor Project
This project uses different models to predict diabetes in a patient using the data from the Predict Diabetes dataset on Kaggle.

## Objective

To predict diabetes in a patient with maximum recall score.

## Features
1.**Pregnancies**: the number of pregnancies prior to collecting the sample <br>
2.**Glucose**: the plasma glucose concentration at 2 hours in an oral Glucose Tolerance Test <br>
3.**BloodPressure**: the diastolic blood pressure measured in mm Hg, 0 indicates a missing value <br>
4.**SkinThickness**: the tricep skin fold thickness, measured in mm, 0 indicates a missing value <br>
5.**Insulin**: the 2 hour serum insulin measured in microunits per milliliter <br>
6.**BMI**: the Body Mass Index measured in kg/m<sup>2</sup> <br>
7.**DiabetesPedigreeFunction**: a value the authors of the paper<sup>2</sup> defined which uses information about the sample'srelatives' genetic background to determine the influence genetics play in a patient developing diabetes <br>
8.**Age**: the age in years<br>

## Target

**Outcome**: a value of 0 is negative for diabetes and a value of 1 is positive for diabetes.

## Methods

* Prior to splitting the data, I used the log(1 + X) transform to transform the following features with skewed distributions: **1**, **5**, **7**, and **8**. <br>
* The first model consisted of a pipeline whose first step was a KNNImputer to impute the missing values of **3** and **4**, and whose second step was a random forest classifier. GridSearchCV was used to find the best hyperparameters with recall as the scoring metric. <br>
* The second model consisted of a pipeline whose first step was a KNNImputer to impute the missing values of **3** and **4**, followed by a MinMaxScaler, then a support vector classifier. GridSearchCV was used in the same fashion as above. <br>
* For both models, I set the class weight to "balanced" to effectively oversample the minority class (1 being the minority). <br>

## Results

I was surprised by my results, in which the support vector machine outperformed the random forest. The hyperparameters on the best support vector classifier were C=100 and gamma = 0.1, which yielded a recall score of 0.7611940298507462.
The hyperparameters on the best random forest classifier were n_estimators = 80 and max_depth = 4, which yielded a recall score of 0.7313432835820896. In the future it would be interesting to see how an XGBoost classifier performed.

Sources:
1. https://www.kaggle.com/datasets/whenamancodes/predict-diabities

2. Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus
Proc Annu Symp Comput Appl Med Care. 1988 Nov 261-265. PMCID: PMC2245318.

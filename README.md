# DiabetesPredictorProject
This project uses different models to predict diabetes in a patient (using the data from the Predict Diabetes dataset on Kaggle).

The dataset is stored in the diabetes.csv file.  This data includes the following features: Pregnancies, Glucose, BloodPressure,
SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.  The target is the Outcome.  The data on Kaggle originally comes from
the paper "Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus", which I used to find more specific information
about the input data.

Pregnancies is the number of pregnancies each sample had gone through prior to collecting the sample.
Glucose is the plasma glucose concentration at 2 hours in an oral Glucose Tolerance Test.
Blood Pressure is the diastolic blood pressure of the sample measured in mm Hg.  Several samples had a value of 
0 here, which I interpreted to mean that this feature was missing from the sample.
SkinThickness is the tricep skin fold thickness of the sample, measured in mm.  Similar to BloodPressure, many 
samples had a value of 0 here, which I interpreted to mean that this feature was missing from the sample.
Insulin is the 2 hour serum insulin measured in microunits per milliliter.
BMI is the sample's Body Mass Index measured in kg / m^2.
DiabetesPedigreeFunction is a function the authors of the paper defined which uses information about the sample's
relatives' genetic background to determine the influence genetics play in a patient developing diabetes.
Age is the age of the sample in years.
Outcome is a 0 or 1, where 0 indicates the sample does not have diabetes and 1 indicates the sample does have diabetes.

After examining the data, I noticed that the features Insulin, Age, DiabetesPedigreeFunction, and Pregnancies all had 
skewed distributions, so I applied the transform log(1 + X) to each feature to obtain distributions which were closer to
normal.  There was no risk of leakage since this could be performed on any new data obtained after training.  The remaining
features were closer to being normally distributed, so I left them alone.

For the splitting of the training and testing data, since the target was imbalanced with a ratio of approximately 3:2 0 to 1 outcomes,
I chose to stratify by the outcome (y) so as to maintain this ratio in the training and testing datasets.

Given that the number of samples missing in SkinThickness was close to one quarter of the data, and upon graphical inspection, in the 
absence of the missing values, the distribution appeared to be approximately normal (by graphical inspection), I decided to keep this feature 
but impute the missing values.  Similarly, the BloodPressure distribution looked approximately normal in the absence of the missing values, so 
I chose to keep this feature.

For both models, I needed to impute the missing values, so this was the first step in the pipeline.  For the Random Forest Model, since the 
scale of each feature does not play a big role, I did not add any other steps to the pipeline before the classifier.  Since the feature scales in 
the Support Vector Machine are more important to model performance, I added the MinMaxScaler to the pipeline after the KNNImputer and before the
classifier.  In both classifiers, I set the class weights to balanced to effectively oversample the minority class (1 being the minority).
After trying different combinations of hyperparameters in each model, I was surprised to find the support vector machine outperformed the random forest
classifier.  The hyperparameters on the best support vector classifier were C=100 and gamma = 0.1, which yielded a recall score of 0.7611940298507462.
The hyperparameters on the best random forest classifier were n_estimators = 80 and max_depth = 4, which yielded a recall score of 0.7313432835820896.

Sources:
1. https://www.kaggle.com/datasets/whenamancodes/predict-diabities

2. Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus
Proc Annu Symp Comput Appl Med Care. 1988 Nov 261-265. PMCID: PMC2245318.

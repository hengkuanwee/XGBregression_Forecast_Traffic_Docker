# MACHINE LEARNING PIPELINE (PREDICT ACTIVE USERS FOR E-SCOOTER RENTAL)

This is a python project submission for AI Apprenticeship Programme Technical Assessment.

This project is for the prediction of active users ('guest-users' and 'registered-users') for an e-scooter rental service in a city.

The following features are provided:

Independent features:
**Independent Features:** 
* `date`​: Date in YYYY-MM-DD
* `hr`​: Hour (0 to 23) 
* `weather`​: Description of the weather conditions for that hour 
* `temperature`​: Average temperature for that hour (Fahrenheit)
* `feels-like-temperature`​: Average feeling temperature for that hour (Fahrenheit)
* `relative-humidity`:​ Average relative humidity for that hour. Measure of the amount of water in the air (%)
* `windspeed`​: Average speed of wind for that hour
* `psi`:​ Pollutant standard index. Measure of pollutants present in the air. (0 to 400)
 
**Target Features:**
* `guest-users`​: Number of guest users using the rental e-scooters in that hour
* `registered-users`​: Number of registered users using the rental e-scooters in that hour

data url: https://aisgaiap.blob.core.windows.net/aiap6-assessment-data/scooter_rental_data.csv

## Table of Content
1) Overview of the machine learning pipeline
2) Running of the machine learning pipeline
3) Configure your own machine learning pipeline!

### 1) Overview of the machine learning pipeline

#### Step a) Data-preprocessing
After the data is imported, the data is preprocessed based on our findings from exploratory data analysis. (file: eda.ipynb)

#### Step b) Splitting into train and test set
The data is then split into training set X and test set y. 

#### Step c) Encoding categorical features
After step 1, 'weather' is the only categorical feature remaining. Due to the nature of the feature (i.e. not ordinal and only has a few unique values), one hot encoder was used for the encoding process.

#### Step d) Normalization/Scaling of the data
MinMaxScaler was used as it preserves the shape of the dataset.

#### Step e) Training on Machine learning model
Multiple models were trained using GridSearchCV find the model that scored the best on "r2 - Coefficient of determination".

Supervised Learning Regression Models used:
'LinearRegression' - Standard OLS 
'LassoRegression' - OLS with regularization (introduce penalty = absolute of the maginitude of the coefficient)
'RidgeRegression' - OLS with regularization (introduce penalty = square of the maginitude of the coefficient)
'XGBRegression' - gradient boosted decision tree (objective function with training loss and regularization)

#### Step f) Results
The machine learning pipelin will provide you with the following results
- Model performance table (on the training set)
Table tabulating each model being trained, its performance based on scoring selected, and the best parameters that returned the scoring.

- Prediction report (on the test set)
Adj R-squared and Variance between prediction results and test set

- First 30 predictions (on the test set)

### 2) Running of the machine learning pipeline

Machine Learning model created in with Python version 3.6.7/3.6.8 and bash script.

##### Installing Dependencies
Paste the following command on your bash terminal to download dependencies
```
pip install -r requirements.txt
```
##### Running the Machine Learning Pipeline
Paste the following command on your bash terminal to grant permission to execute the 'run.sh' file
```
chmod +x run.sh
```
Paste the following command on your bash terminal to run the machine learning programme
```
./run.sh
```

### 3) Configure your own machine learning pipeline!
A configuration file (file: ./mlp/config.py) was included to allow anyone to make their own configuration to the pipline. Users can make their own configurations to steps(b-e) mentioned above.
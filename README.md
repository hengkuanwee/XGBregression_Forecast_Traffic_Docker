# XGBoost Regression Model to predict traffic volume

This project is for prediction of westbound 'traffic_volume' with the following attributes:

`holiday`​: US national and regional holidays 
`temp`​: average temperature in Kelvin (K) 
`rain_1h`​: rain that occured in the hour (mm) 
`snow_1h`​: snow that occured in the hour (mm) 
`clouds_all`​: percentage of cloud cover 
`weather main`:​ textual description of current weather 
`weather_description`​: longer textual description of current weather 
`date_time`:​ hour of the data collected in local time 

Data source: MN Department of Transportation

## Project segments
The project consists of 2 segments:
1) Exploratory Data Analysis 
2) XGBoost Regression Model to predict traffic volume

## Getting Started
###1) Exploratory Data Analysis

File:
'eda.ipynb'

Created in an interactive notebook in Python. Provides an anaysis of the data provided and its implications. 

#### Running the Exploratory Data Analysis
Open the file in Jupyter Noteborequirements.txt'
'mlp/xgbregressor_ml.py'
'run.sh'ok.

###2) XGBoost Regression Model to predict traffic volume

File:
'aiap_traffic.sh'
'aiap_traffic.yml'
'mysql_password.txt'
'mysql_user.txt'

The data and model has been packaged into Docker image and stored onto docker official repo. In order to run the model, ensure that Docker is installed.

##### Running the Machine Learning Model
Paste the following command on your bash terminal to grant permission to execute the aiap_traffic.sh file
```
chmod +x aiap_traffic.sh
```
Paste the following command on your bash terminal to run the machine learning programme
```
chmod +x ./aiap_traffic.sh
```
Paste the following command on your bash terminal to observe the machine learning model. (ctrl-c to exit)
```
docker container logs -f myapp_aiap_mlmodel
```

#### Expected Results ()
The machine learning model will provide you with the following results
1) Model report
Accuray and the best parameters from grid_search

2) Prediction report
Adj R-squared and Variance between prediction results and test set

3) First 30 predictions 

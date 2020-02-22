# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:27:13 2020

@author: kuanw
"""
# System
import io, os, sys, datetime, math, calendar, time

# Data Manipulation
import numpy as np
import pandas as pd

# Data Preprocessing
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score,precision_score,f1_score,r2_score,explained_variance_score
from xgboost import XGBClassifier, XGBRegressor

# Visualisation
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sn

# MySQL
import mysql.connector
from mysql.connector import Error

# Import Data from MySQL "aiap_traffic" database
def MySQL_Connection(host, mysql_user, mysql_password, port, database, table):
	config = {
		'host': host,
		'user': mysql_user,
		'password': mysql_password,
		'port': port,
		'database': database
	}
	
	tries = 30
	while tries > 0:
		try:
			connection = mysql.connector.connect(**config)
			if connection.is_connected():
				db_Info = connection.get_server_info()
				print("\nConnected to MySQL Server version ", db_Info)
				dataset = pd.read_sql("SELECT * FROM " + table, con=connection)
				print(database + " is extracted." )   
				break  
		except:
			print("Waiting for MySQL database to be set up...")
			time.sleep(10)
		finally:
			tries -= 1
	
	if connection.is_connected():
		return dataset
		connection.close()
		print("MySQL connection is closed")
		print("Done")
	else:
		print("MySQL connection failed")

# Drop features
def drop_features (dataset, column, axis=1):
	dataset.drop([column], axis = axis)
	
	return dataset
	
# Create the additional features "hours_before_holiday" and "hours_after_holiday", and remove "holiday"
def add_features_holiday (dataset, column, hours=24):
	# Create blank list for hours_before and hours_after, if the row is within 24 hours from a holiday, we will append the row number to it
	hours_before = []
	hours_after = []
	# Create blank list for hours_holiday, if the row is the holiday itself, we will append the row number to it
	hours_holiday = []
	# Create numpy arrays of False, if row number is within 24 hours from a holiday, we will change it to True
	before_holiday = np.zeros(len(dataset[column])).astype(dtype=bool)
	after_holiday = np.zeros(len(dataset[column])).astype(dtype=bool)
	for index, row in dataset[column].to_frame().iterrows():
		# If there is a holiday, append the relevant number to hours_holiday
		if row[column] != "None":
			hours_holiday.append(index)
	
	# Append the relevant row humbers to hours_before and hours_after
	for i in hours_holiday:
		for hour in range(0, hours+1):
			hours_before.append(i - hour)
			hours_after.append(i + hour)
			
	# Remove the row rumbers that are out of range
	hours_before = np.asarray(hours_before)
	hours_before = hours_before[(hours_before>=0) & (hours_before<=len(dataset[column]))]
	hours_after = np.asarray(hours_after)
	hours_after = hours_after[(hours_after>=0) & (hours_after<=len(dataset[column]))]
	
	# Change numpy array to true, if the respective row number within 24 hours from a holiday
	before_holiday[hours_before.tolist()] = True
	after_holiday[hours_after.tolist()] = True
	
	# Convert hours_before_holiday and hours_after_holiday to dataframe and merge to original dataset
	before_holiday = pd.DataFrame(before_holiday,  columns=['before_holiday'])
	after_holiday = pd.DataFrame(after_holiday,  columns=['after_holiday'])
	
	dataset =  pd.concat([dataset, before_holiday], axis=1, sort=False)
	dataset =  pd.concat([dataset, after_holiday], axis=1, sort=False)
	
	# Drop column as relevant features were already extracted and feature takes into account column
	dataset = dataset.drop([column], axis = 1)
			
	return dataset

def convert_datetime_format(dataset, column):
	dataset[column] = pd.to_datetime(dataset[column], format="%Y-%m-%d %H:%M:%S")
	
	return dataset

# Create the additional features "year", "month", "day_of_the_week" and "time_period", and remove "date_time"
def add_features_datetime_YMD (dataset, column="date_time", feature_name=["year", "month", "day", "time"]):
	# Create numpy arrays of zeros/empty string, we will replace the values subsequently
	dt_year = np.ones(len(dataset[column]))
	dt_month = np.ones(len(dataset[column]))
	dt_day = []
	dt_time = np.ones(len(dataset[column]))
	
	# Extract the relevant feature from column and update the features to dataset
	for feature in feature_name:
		if feature == "year":
			for index, row in dataset[column].to_frame().iterrows():
				dt_year[index] = row[column].year
			dt_year = pd.DataFrame(data=dt_year, columns=['year'], dtype=np.int64)
			dataset =  pd.concat([dataset, dt_year], axis=1, sort=False)
		elif feature == "month":
			for index, row in dataset[column].to_frame().iterrows():
				dt_month[index] = row[column].month
			dt_month = pd.DataFrame(data=dt_month, columns=['month'], dtype=np.int64)
			dataset =  pd.concat([dataset, dt_month], axis=1, sort=False)
		elif feature == "day":
			for index, row in dataset[column].to_frame().iterrows():
				dt_day.append(row[column].strftime('%A'))
			dt_day = pd.DataFrame(data=dt_day, columns=['day_of_the_week'], dtype=str)
			dataset =  pd.concat([dataset, dt_day], axis=1, sort=False)
		elif feature == "time":
			for index, row in dataset[column].to_frame().iterrows():
				dt_time[index] = row[column].hour
			dt_time = pd.DataFrame(data=dt_time, columns=['time_period'], dtype=np.int64)
			dataset =  pd.concat([dataset, dt_time], axis=1, sort=False)
	
	# Drop column as relevant features were already extracted
	dataset = dataset.drop([column], axis = 1)
			
	return dataset

# Classify time period into bins of Morning, Afternoon, Evening and Night. For each bin, the traffic is expected to be different
def time_period_bin(dataset, column):
	dataset[column] = pd.cut(dataset[column], 
									bins=[0,6,12,18,23], 
									labels=['Night','Morning','Afternoon','Evening'],
									include_lowest=True)
	return dataset

# Encoding categorical data (i.e. Creating dummy variables)
# Label encode each categorical variable
def labelencode (dataset, columns = [0]):
	for column in columns:
		labelencoder_data = LabelEncoder()
		dataset[:,column] = labelencoder_data.fit_transform(dataset[:,column])
	return dataset

# Gridsearch to find the best model parameters
def XGBRegressor_gridsearch(X_train, y_train, X_test, y_test):
	xgb_model = XGBRegressor(objective='reg:squarederror', 
							  tree_method='exact', 
							  early_stopping_rounds = 50)
							  
	parameters = {'max_depth': [9], 
					'learning_rate': [0.015], 
					'n_estimators': [1500], 
					'gamma': [0], 
					'min_child_weight': [1], 
					'subsample': [0.8], 
					'colsample_bytree': [0.9], 
					#'seed': [10]
				 }
				 
	clf = GridSearchCV(xgb_model, parameters, n_jobs=-1, 
					   cv=10,
					   verbose=0, refit=True)
	
	clf.fit(X_train, y_train)
	
	print("Done")
	
	# Print model report:
	print ("\n"+ "\033[1m" + "Model Report" + "\033[0m")
	print("Best: Accuracy of %f using: \n %s" % (clf.best_score_, clf.best_params_))
	
	# Predicting the test set results
	y_pred = clf.predict(X_test).astype(int)
	
	# Print prediction report on test set
	print ("\n" + "\033[1m"+ "Prediction Report" + "\033[0m")
	print("Adj R-squared : " + str(r2_score(y_test,y_pred)))
	print("Variance: " + str(explained_variance_score(y_test,y_pred)))
	asd = np.squeeze(y_test)
	
	# Print predictions
	predictions = np.concatenate([(np.squeeze(y_test), y_pred)]).T
	predictions = pd.DataFrame(predictions, columns=["traffic_volume_test", "traffic_volume_pred"])
	
	print ("\n" + "\033[1m"+ "First 30 predictions" + "\033[0m")
	print(predictions.head(30))

if __name__ == '__main__':
	print ("\n" + "\033[1m"+ "Importing data..." + "\033[0m")
	
	host = 'db'
	mysql_user = open("/run/secrets/mysql_user", "r") .read() 
	mysql_password = open("/run/secrets/mysql_password", "r") .read() 
	port = 3306
	database = 'aiap_traffic'
	table = 'traffic_data'
	
	dataset = MySQL_Connection(host, mysql_user, mysql_password, port, database, table)
	
	print ("\n" + "\033[1m"+ "Conducting feature cleaning and engineering..." + "\033[0m")
	
	# Pre-processing of data - steps identified from EDA
	# 1) Drop "snow_1h"
	print ("step 1 - drop 'snow_1h'")
	dataset = drop_features(dataset, "snow_1h")

	# 2) Create the additional features "hours_before_holiday" and "hours_after_holiday", and remove "holiday"
	print ("step 2 - add features for 'holiday'")
	dataset = add_features_holiday(dataset, "holiday")

	# 3) Drop "weather_main"
	print ("step 3 - drop 'weather_main'")
	dataset = drop_features(dataset, "weather_main")

	# 4) Convert date_time to date_time datatype
	print ("step 4 - convert 'date_time' format")
	dataset = convert_datetime_format(dataset, "date_time")

	# 5) Create the additional features "year", "month", "day_of_the_week" and "time_period", and remove "date_time"
	print ("step 5 - add features for 'date_time'")
	dataset = add_features_datetime_YMD (dataset, column="date_time", feature_name=["year", "month", "day", "time"])
	
	# 6) Classify time period into bins of Morning, Afternoon, Evening and Night. For each bin, the traffic is expected to be different
	print ("step 6 - create 'time_period' bin")
	dataset = time_period_bin(dataset, "time_period")

	# 7) Split into X and y
	print ("step 7 - Split into X and y")
	X = dataset.drop(["traffic_volume"], axis = 1).iloc[:, :].values
	y = dataset.loc[:,"traffic_volume"].values

	# 8) Labelencode and OneHotEncode (drop a dummy variable to prevent dummy variable trap (if required))
	print ("step 8 - Labelencode and OneHotEncode")
	X = labelencode(dataset=X, columns=[4,5,6,7,8,9,10,11] )
	ct_1 = ColumnTransformer(
			[('one_hot_encoder', OneHotEncoder(drop='first', categories='auto'), [3,6,7,8,9])],
			remainder = 'passthrough')
	X = ct_1.fit_transform(X).toarray()
	
	# Splitting the dataset into the Training set and Test set
	print ("Step 9 - Split into Training set and Test set")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
	
	# Conduct GridSearch, to find the optimal hyper-parameters
	print ("\n" + "\033[1m"+ "Training XGBRegressor Model..." + "\033[0m")
	XGBRegressor_gridsearch(X_train, y_train, X_test, y_test)

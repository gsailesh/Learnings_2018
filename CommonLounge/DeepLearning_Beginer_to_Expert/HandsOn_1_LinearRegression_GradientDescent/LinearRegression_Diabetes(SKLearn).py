
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import Diabetes dataset
diabetes_data = datasets.load_diabetes()
diabetes_X = diabetes_data.data
diabetes_y = diabetes_data.target

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Linear Regression model
model = LinearRegression()
# Train/Fit the model 
model.fit(diabetes_X_train, diabetes_y_train)
# Evaluate the model
diabetes_y_predict = model.predict(diabetes_X_test)

model_mse = mean_squared_error(diabetes_y_test, diabetes_y_predict)
print (model_mse)
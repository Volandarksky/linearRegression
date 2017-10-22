import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
dataframe = pd.read_fwf('linear_regression_dataset.txt')
x_values = dataframe[['x']]
y_values = dataframe[['y']]

y_reg = linear_model.LinearRegression()
y_reg.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, y_reg.predict(x_values))
plt.show()

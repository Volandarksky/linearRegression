import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
dataframe = pd.read_fwf('animals_brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from arima_function import arima_forecast



# read time series from the euro-dollar-exchange.csv file
exchange_series = pd.read_csv('euro-dollar-exchange.csv', index_col=0, skiprows=15)


# View 5 records
print(exchange_series.head(5))


# View some stats information
print(exchange_series.describe())

show_plot = exchange_series.plot()
plt.show()

# Size of exchange rates
nr_of_elem = len(exchange_series)

# Use 70% of data as training, rest 30% to Test model
training_size = int(nr_of_elem * 0.7)
training_data = exchange_series[0:training_size]
test_data = exchange_series[training_size:nr_of_elem]

# new arrays to store actual and predictions
actual = [x for x in training_data.values]
predictions = list()

# in a for loop, predict values using ARIMA model
for timepoint in range(len(test_data)):
    actual_val = float(test_data.values[timepoint])
    # forcast value
    prediction = arima_forecast(actual, 3, 1, 0)
    print('Actual Value=%f, Predicted Value=%f' % (actual_val, prediction))
    # add it in the list
    predictions.append(prediction)
    actual.append(actual_val)

# Display MSE to see how good the model is
error = mean_squared_error(test_data, predictions)
print('Test MSE(smaller the better fit): %.3f' % error)
# plot
plt.plot(test_data)
plt.plot(predictions, color='red')
plt.show()

# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np  # to work with arrays
                    # only numpy arrays can be input to neural networks when we work with keras
import matplotlib.pyplot as plt  # to visualize the data
import pandas as pd  # to work with dataframes

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
                    # iloc method is used to get the right columns from the dataset that is used 
# Feature Scaling
                    # there are generally two ways of feature scaling : one is standardization and other is normalization
                    # here it is recommended to use normalization since we are using sigmoid function in output layer
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))  # all the scaled feature stock price will be between 0 and 1
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# 60 timesteps means rnn will learn from the 60 days of stock price from time t in order to generate the output at time t 

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential  # class that will allow us to create a neural network objects having a sequence of layers
from keras.layers import Dense  # class to add output layer
from keras.layers import LSTM  # class to add lstm layers
from keras.layers import Dropout  # class to add some dropout regularization

# Initialising the RNN
regressor = Sequential()    # regressor represents a instance of sequential class that represents a sequence of layers

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # it means 20% of the neurons in the lstm layer will be ignored while training the model

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1)) # units represents the number of neurons that should be there in the output layer

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')  # loss represents the cost function
                                                                    # adam represents stochastic gradient descent algorithm
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)  # for vertical concatenation we use axis = 0, and for horizontal concatenation axis = 1
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

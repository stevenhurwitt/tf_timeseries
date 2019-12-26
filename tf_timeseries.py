import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

print('reading data...')
print('')
apple = pd.read_csv('AAPL.csv')
apple.head()

apple.set_index('Date', drop = True, inplace = True)


apple.plot(y = 'Open')


# Preprocessing
print('processing data...')
print('')
apple_training_processed = apple.iloc[:, 0:1].values
apple_training_processed.shape


scaler = MinMaxScaler(feature_range = (0, 1))

apple_training_scaled = scaler.fit_transform(apple_training_processed)
apple_training_scaled.shape


features_set = []
labels = []

for i in range(60, 1510):
    features_set.append(apple_training_scaled[i-60:i, 0])
    labels.append(apple_training_scaled[i, 0])


features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))


# Train LSTM
# LSTM with four layers of 50 units, dense layer at end to predict one value

print('creating model...')
print('')
model = Sequential()


# In[18]:


model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))


model.compile(optimizer = 'adam', loss = 'mean_squared_error')

print('compiling & fitting model...')
print('')
model.fit(features_set, labels, epochs = 29, batch_size = 60)


# Test Model

print('reading test data...')
print('')
apple_test = pd.read_csv('AAPL_test.csv')
apple_test.set_index('Date', drop = True, inplace = True)
apple_testing_processed = apple_test.iloc[:, 0:1].values


apple_test.plot(y = 'Open')


apple_total = pd.concat((apple['Open'], apple_test['Open']), axis=0)

test_inputs = apple_total[len(apple_total) - len(apple_test) - 60:].values

print('processing test data...')
print('')
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)


test_features = []
for i in range(60, 80):
    test_features.append(test_inputs[i-60:i, 0])


test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))


# ## Make Predictions

print('making predictions...')
print('')
predictions = model.predict(test_features)


predictions = scaler.inverse_transform(predictions)

print('plotting data...')
print('')
plt.figure(figsize=(10,6))
plt.plot(apple_testing_processed, color='blue', label='Actual Apple Stock Price')
plt.plot(predictions , color='red', label='Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()

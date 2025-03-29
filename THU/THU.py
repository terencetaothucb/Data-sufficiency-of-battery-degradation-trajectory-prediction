#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split

def time_series_to_supervised(data, n_in=1, n_out=1,dropnan=True):
   
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    origNames = df.columns
    cols, names = list(), list()
    cols.append(df.shift(0))
    names += [('%s' % origNames[j]) for j in range(n_vars)]
    n_in = max(0, n_in)
    for i in range(n_in, 0, -1):
        time = '(t-%d)' % i
        cols.append(df.shift(i))
        names += [('%s%s' % (origNames[j], time)) for j in range(n_vars)]
    n_out = max(n_out, 0)
    for i in range(1, n_out+1):
        time = '(t+%d)' % i
        cols.append(df.shift(-i))
        names += [('%s%s' % (origNames[j], time)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


file_path = r'Readme-Data.xlsx'  
df = pd.read_excel(file_path, sheet_name='Exp-1')

battery_types = ['B26T55', 'B1T25']
feature_names=['VC89','VD89','tVD89','ReVC','ReVD','tReVD','Vg1','Vg2','Vg3','Vg4','Vg5','Vg6','Vg7','Vg8','Vg9','RVg',
               'Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9',
               'RL1','RL2','RL3','RL4','RL5','RL6','RL7','RL8','RL9',
               'RO1','RO2','RO3','RO4','RO5','RO6','RO7','RO8','Capacity']
B26T55 = df[df['Battery_Name'] == 'B26T55']
B1T25 = df[df['Battery_Name'] == 'B1T25']

feature_26_T55=pd.concat([B26T55['Vg1'],B26T55['Vg9'],B26T55['RVg'],
                          B26T55['Q1'],B26T55['Q2'],B26T55['Q3'],B26T55['Q4'],B26T55['Q5'],B26T55['Q6'],B26T55['Q7'],B26T55['Q9'],
                          B26T55['RL1'],B26T55['RL8'],B26T55['RL9'],
                          B26T55['RO8']],axis=1)

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_26_T55 = scaler.fit_transform(feature_26_T55)
scaledFeature_26_T55 = pd.DataFrame(data=scaledFeature_26_T55)

print(scaledFeature_26_T55.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_26_T55 = time_series_to_supervised(scaledFeature_26_T55,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_26_T55 = scaler.fit_transform(pd.DataFrame(B26T55['Capacity']))

n_steps_in =3 
n_steps_out=1
processedCapacity_26_T55 = time_series_to_supervised(scaledCapacity_26_T55,n_steps_in,n_steps_out)

data_x26 = processedFeature_26_T55.loc[:,'0(t-3)':'14(t-1)']
data_y26=processedCapacity_26_T55.loc[:,'0']
data_y26=data_y26.values.reshape(-1,1)
train_X26=data_x26.values[:899]
train_y26=data_y26[:899]
train_X26 = train_X26.reshape((train_X26.shape[0], n_steps_in, 15))


# In[ ]:


n_steps_in =3 
n_steps_out=1
processedFeature_26_T55 = time_series_to_supervised(scaledFeature_26_T55,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_26_T55 = scaler.fit_transform(pd.DataFrame(B26T55['Capacity']))
soure_data=899

processedCapacity_26_T55 = time_series_to_supervised(scaledCapacity_26_T55,n_steps_in,n_steps_out)
data_x26 = processedFeature_26_T55.loc[:,'0(t-3)':'14(t-1)']
data_y26=processedCapacity_26_T55.loc[:,'0']
data_y26=data_y26.values.reshape(-1,1)
train_X26=data_x26.values[:soure_data]
train_y26=data_y26[:soure_data]
train_X26 = train_X26.reshape((train_X26.shape[0], n_steps_in,15))


# In[ ]:


# Train Source model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from tensorflow.keras.models import save_model, load_model
import os
import time
# Set random seed
np.random.seed(42)
tf.random.set_seed(42)
# Start time
start_time=time.time()
# Initialize the model
model = Sequential()
model.add(LSTM(96, return_sequences=True, input_shape=(train_X26.shape[1], train_X26.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
# Compile the model
model.compile(loss='mse', optimizer='adam')
print(model.summary())

# Training
history = model.fit(train_X26, train_y26, epochs=75, batch_size=64,verbose=2, shuffle=False)
# End time
end_time = time.time()
# Running time
elapsed_time = end_time - start_time
# Print model running time
print(f"Model running time：{elapsed_time} seconds")


# In[ ]:


# No Finetuning 
train_samples=300
import matplotlib
feature_1_T25=pd.concat([B1T25['Vg1'],B1T25['Vg9'],B1T25['RVg'],
                         B1T25['Q1'],B1T25['Q2'],B1T25['Q3'],B1T25['Q4'],B1T25['Q5'],B1T25['Q6'],B1T25['Q7'],B1T25['Q9'],
                         B1T25['RL1'],B1T25['RL8'],B1T25['RL9'],B1T25['RO8']],axis=1)


B1T25.reset_index(drop=True, inplace=True)
scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_1_T25 = scaler.fit_transform(feature_1_T25)

scaledFeature_1_T25 = pd.DataFrame(data=scaledFeature_1_T25)

n_steps_in =3 
n_steps_out=1
processedFeature_1_T25 = time_series_to_supervised(scaledFeature_1_T25,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_1_T25 = scaler.fit_transform(pd.DataFrame(B1T25['Capacity']))
processedCapacity_1_T25 = time_series_to_supervised(scaledCapacity_1_T25,n_steps_in,n_steps_out)

data_x1 = processedFeature_1_T25.loc[:,'0(t-3)':'14(t-1)']
data_y1=processedCapacity_1_T25.loc[:,'0']
data_y1=data_y1.values.reshape(-1,1)

train_X1=data_x1.values[:train_samples]
test_X1=data_x1.values[train_samples:]
train_y1=data_y1[:train_samples]
test_y1=data_y1[train_samples:]
train_X1 = train_X1.reshape((train_X1.shape[0], n_steps_in, 15))
test_X1 = test_X1.reshape((test_X1.shape[0], n_steps_in, 15))

yhat1 = model.predict(test_X1)
test_y1=test_y1.reshape(-1,1)
inv_forecast_y1 = scaler.inverse_transform(yhat1)
inv_test_y1 = scaler.inverse_transform(test_y1)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

mape_1 = mape(inv_test_y1, inv_forecast_y1)
print('Test MAPE: %.3f' % mape_1)

plt.figure(figsize=(8,6))
plt.plot(B1T25['Capacity'], label='True')
x_range = range(train_samples, train_samples + len(inv_forecast_y1))
plt.plot(x_range,inv_forecast_y1,marker='.',label='LSTM',linestyle=None,markersize=5)
plt.xlim(0,1400)
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.annotate('prediction starting point(cycle=train_samples)',xy=(train_samples, B1T25['Capacity'].iloc[train_samples]), xytext=(train_samples, B1T25['Capacity'].iloc[120]),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='black')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[ ]:


# With Finetuning
from keras.models import Model
from tensorflow.keras.layers import Input

# Freeze the first two LSTM layers
for layer in model.layers[:2]:  
    layer.trainable = False

# Create a new output layer
input_layer = Input(shape=(train_X1.shape[1], train_X1.shape[2]))
lstm_output_1 = model.layers[0](input_layer)  
lstm_output_2 = model.layers[1](lstm_output_1)  
new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
#new_dense_1 = model.layers[2](lstm_output_2)
new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)
#new_output_layer = model.layers[3](new_dense_1)

# Create and compile the new model
transfer_model = Model(inputs=input_layer, outputs=new_output_layer)

# Fine-tune the model on the target dataset
transfer_model.compile(loss='mse', optimizer='adam')
transfer_model.fit(train_X1, train_y1, epochs=20, batch_size=64, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)

# Prediction and model evaluation
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

yhat1t = transfer_model.predict(test_X1)
test_y1 = test_y1.reshape(-1, 1)

inv_forecast_y1t = scaler.inverse_transform(yhat1t)
inv_test_y1t = scaler.inverse_transform(test_y1)

mape_1t = mape(inv_test_y1t, inv_forecast_y1t)
print('Test MAPE: %.3f' % mape_1t)
# Visualization
plt.figure(figsize=(8,6))
plt.plot(B1T25['Capacity'], label='True')
x_range = range(train_samples, train_samples + len(inv_forecast_y1t))
plt.plot(x_range,inv_forecast_y1t,marker='.',label='LSTM+Fine-tune',linestyle=None,markersize=5)
plt.xlim(0,1400)
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.annotate('prediction starting point',xy=(train_samples, B1T25['Capacity'].iloc[train_samples]), xytext=(train_samples, B1T25['Capacity'].iloc[120]),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='black')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[1]:


import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

file_path = r'Readme-Data.xlsx'  
df = pd.read_excel(file_path, sheet_name='Exp-2')
end=1301
# List of battery types
battery_types = ['B1T25', 'B2T25', 'B3T25', 'B4T25', 'B5T25', 'B6T25', 'B7T25', 'B8T25', 'B9T25']
feature_names = ['VC89', 'VD89', 'tVD89', 'ReVC', 'ReVD', 'tReVD', 'Vg1', 'Vg2', 'Vg3', 'Vg4', 'Vg5', 'Vg6', 'Vg7', 'Vg8', 'Vg9', 'RVg',
                 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9',
                 'RL1', 'RL2', 'RL3', 'RL4', 'RL5', 'RL6', 'RL7', 'RL8', 'RL9',
                 'RO1', 'RO2', 'RO3', 'RO4', 'RO5', 'RO6', 'RO7', 'RO8']


#  Store the results of average absolute value correlation
avg_abs_corr_results = {}

# Circularly calculate the average absolute value correlation in each battery under different cycles
for battery_type in battery_types:
    battery_data = df[df['Battery_Name'] == battery_type]
    
    # Store the average absolute value correlation result of the current battery
    avg_abs_corr_results[battery_type] = {}
    
    # Compute the average absolute value correlation for each feature
    for feature_name in feature_names:
        # Store the average absolute value correlation result of the current feature
        avg_abs_corr_results[battery_type][feature_name] = []
        
        #  Calculation of average absolute value correlation under different periods
        for period in range(20, end, 20):
            # Get the data of the current period
            current_period_data = battery_data.iloc[:,:-1].head(period)
            
            # Calculate the absolute correlation between the current feature and the capacity
            correlation_result = np.abs(current_period_data[feature_name].corr(battery_data.iloc[:,-1].head(period)))
            #correlation_result = current_period_data[feature_name].corr(battery_data.iloc[:,-1].head(period))
            
            # Store results
            avg_abs_corr_results[battery_type][feature_name].append(correlation_result)

# Calculate the average absolute value correlation of all features for all batteries under each cycle
avg_abs_corr_results['Average'] = {}
for period in range(20, end, 20):
    avg_abs_corr_results['Average'][period] = []
    for feature_name in feature_names:
        # Calculate the average absolute value correlation of all batteries and all features in the current cycle
        avg_corr = np.mean([avg_abs_corr_results[battery_type][feature_name][period // 20 - 1] for battery_type in battery_types])
        avg_abs_corr_results['Average'][period].append(avg_corr)
av_all_25=[]
for period in range(20, end, 20):
    av=np.mean(avg_abs_corr_results['Average'][period])
    av_all_25.append(av)

plt.figure(figsize=(12, 8))
# Plot the average absolute value correlation for all features for each cycle
plt.plot(np.arange(20, end, 20), av_all_25,'o', label=f'Average - All Features')
plt.title('Average Absolute Correlation between All Features and Capacity over Cycles')
plt.xlabel('Cycles')
plt.ylabel('Average Absolute Correlation')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[4]:


from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

file_path = r'Readme-Data.xlsx'  
df = pd.read_excel(file_path, sheet_name='Exp-3')

# Battery type list
battery_types_T55 = ['B26T55', 'B27T55', 'B28T55', 'B29T55', 'B30T55', 'B31T55', 'B32T55']
battery_types_T25 = ['B1T25','B2T25','B3T25','B4T25','B5T25','B6T25','B7T25','B8T25','B9T25']

# The list of features to be analyzed
feature_names = ['VC89', 'VD89', 'tVD89', 'ReVC', 'ReVD', 'tReVD', 'Vg1', 'Vg2', 'Vg3', 'Vg4', 'Vg5', 'Vg6', 'Vg7', 'Vg8',
                  'Vg9', 'RVg', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'RL1', 'RL2', 'RL3', 'RL4', 'RL5',
                  'RL6', 'RL7', 'RL8', 'RL9', 'RO1', 'RO2', 'RO3', 'RO4', 'RO5', 'RO6', 'RO7', 'RO8']

periods = list(range(20, 901, 20))

# A list of W distances for each cycle
wasserstein_distances_all_periods = []

# Traverse the T55 and T25 batteries
for battery_T55 in battery_types_T55:
    for battery_T25 in battery_types_T25:

        # Get the data of the corresponding battery
        df_battery_T55 = df[df['Battery_Name'] == battery_T55]
        df_battery_T25 = df[df['Battery_Name'] == battery_T25]

        # Extract the features of the corresponding battery
        feature_T55 = df_battery_T55[feature_names]
        feature_T25 = df_battery_T25[feature_names]

        # Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        feature_T55_normalized = scaler.fit_transform(feature_T55)
        feature_T25_normalized = scaler.fit_transform(feature_T25)

        wasserstein_distances = []

        # Calculate the W distance under different cycles
        #for feature_idx in range(0, 42):# (0,6)VC VD,(6,16)Vg,(16,25)Q,(25,34)RL,(34,42)RO,(0,1)VC89,(0,42)全部
        for feature_idx in [41]:    
            distances = []
            
            for period in periods:
                # Extract a subset of data
                subset_T25 = feature_T25_normalized[:period, feature_idx]
                subset_T55 = feature_T55_normalized[:period, feature_idx]

                # Calculate W distance
                w_distance = wasserstein_distance(subset_T25.flatten(), subset_T55.flatten())
                distances.append(1-w_distance)

            wasserstein_distances.append(distances)

        # Calculate the sum of distances of multiple features of a pair of batteries under each cycle
        sum_distances = np.sum(wasserstein_distances, axis=0)
        # Calculate the average
        average_distances1 = sum_distances / len(wasserstein_distances)
        # Multiple pairs of batteries, multiple features, and distances under different cycles
        wasserstein_distances_all_periods.append(average_distances1)

# Convert to NumPy array.
wasserstein_distances_np_all_periods = np.array(wasserstein_distances_all_periods)

# Calculate the sum of the distances of all multiple features in multiple pairs of cells per cycle
sum_distances_all_periods = np.sum(wasserstein_distances_np_all_periods, axis=0)

# Calculate the average value and standard deviation
average_distances_all_periods1 = sum_distances_all_periods / len(wasserstein_distances_all_periods)#len(wasserstein_distances_all_periods)是63
std_distances_all_periods1 = np.std(wasserstein_distances_np_all_periods, axis=0)

# Plot Figure
plt.figure(figsize=(5, 3),dpi=600)
plt.plot(periods, average_distances_all_periods1, marker='o')
plt.title('Average TC of Features')
plt.xlabel('Cycles')
plt.ylabel('Transferable Capability')
plt.ylim()
#plt.legend()
plt.grid(True)
plt.show()

# # Plot std
# plt.figure(figsize=(5, 3),dpi=600)
# plt.plot(periods, std_distances_all_periods1, marker='o', label='Standard Deviation of Wasserstein Distance')
# plt.title('Standard Deviation of Wasserstein Distance between Vg Features of T55 and T25 Batteries')
# plt.xlabel('Number of Cycles to Caculate W Distance')
# plt.ylabel('Standard Deviation of Wasserstein Distance')
# plt.ylim(0, 0.2)
# plt.legend()
# plt.grid(True)
# plt.show()


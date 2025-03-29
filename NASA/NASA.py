#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load Battery Data
import numpy as np
import scipy.io
from datetime import datetime
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')

# Convert time format, convert string to datatime format
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# Load mat files
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split('/')[-1].split('.')[0]
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        
        for j in range(len(k)):
            t = col[i][3][0][0][j][0];
            l = [t[m] for m in range(len(t))]
            d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data

# Battery capacity
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]

# Battery impedance
def getBatteryImpedance(Battery):
    cycle, impedance = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'impedance':
            impedance.append(Bat['data']['Re'][0])
            cycle.append(i)
            i += 1
    return [cycle, impedance]

# Battery test data for charging or discharging 
def getBatteryValues(Battery, Type='charge'):
    data=[]
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018'] 
dir_path = r'NASA'

capacity,impedance,charge,discharge ={},{},{},{}

for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    capacity[name] = getBatteryCapacity(data)              # Capacity data during discharging
    charge[name] = getBatteryValues(data, 'charge')        # Charging data
    discharge[name] = getBatteryValues(data, 'discharge')  # Discharging data
    impedance[name] = getBatteryImpedance(data)


# In[3]:


# Voltage-SOC
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

name = 'B0007'  
time = range(2, 168)  

fig, ax = plt.subplots(1, figsize=(4, 3),dpi=600)

cmap = cm.get_cmap('Blues', len(time)) 
colors = cmap(np.linspace(0, 1, len(time)))

invalid_cycles = []

for idx, t in enumerate(time):
    Battery = charge[name][t]  

    voltage = np.array(Battery['Voltage_measured'])  
    capacity = np.array(Battery['Time']) 

    if np.any(voltage > 5):
        invalid_cycles.append(t)
        continue  

    closest_idx = np.argmin(np.abs(voltage - 4.19))

    voltage = voltage[:closest_idx + 1]
    capacity = capacity[:closest_idx + 1]

    ax.plot(capacity, voltage, color=colors[idx], label=f'Cycle {t}' if t % 20 == 0 else "")  
sm = cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=time.start, vmax=time.stop - 1))
#plt.colorbar(sm, label='Cycle Number')

plt.title('Voltage vs. Capacity for Cycles', fontsize=20)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Voltage (V)', fontsize=20)

if invalid_cycles:
    print(f"Cycles with voltage exceeding 5V: {invalid_cycles}")

plt.ylim(2,4.25)
plt.tight_layout()
plt.show()


# In[8]:


# capacity
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3), dpi=600)

plt.plot(capacity['B0005'][1], marker='o', markersize=3, color=plt.cm.Reds(0.8), label='Battery B0005')

plt.plot(capacity['B0007'][1], marker='o', markersize=3, color=plt.cm.Blues(0.8), label='Battery B0007')

plt.title('Capacity Decay of Battery B0005 and B0007', fontsize=14)
plt.xlabel('Cycle', fontsize=12)
plt.ylabel('Capacity (mAh)', fontsize=12)

plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'Feature_5.xlsx'  
Feature_5 = pd.read_excel(file_path).iloc[1:168,1:]
Capacity_5 = pd.DataFrame(capacity['B0005'][1]).iloc[1:168, :]

# Create an empty list to store the correlation trend between each feature and capacity over cycles
correlation_trend_5 = []
periods = range(5, 161, 5)

# Loop to calculate the correlation between each feature and capacity over different cycles
for col, feature_data in Feature_5.items():
    correlations_5 = []  # Store correlation values for each cycle

    for seg in periods:
        # Slice data based on the cycle period
        feature_slice = feature_data.iloc[:seg]
        capacity_slice = Capacity_5.iloc[:seg, 0]  # Capacity has only one column

        # Compute correlation
        correlation = feature_slice.corr(capacity_slice)

        # Add result to the list
        correlations_5.append(abs(correlation))

    # Append the correlation trend of each feature with capacity to the overall list
    correlation_trend_5.append(correlations_5)

# Convert to a NumPy array for further processing
correlation_trend_5 = np.array(correlation_trend_5)

# Compute the average correlation for each cycle
average_correlations_5 = np.mean(correlation_trend_5, axis=0)

# Plot the correlation trend of each feature with capacity over cycles
plt.figure(figsize=(10, 6))
for i in range(correlation_trend_5.shape[0]):
    plt.plot(periods, correlation_trend_5[i], label=f'Feature {i+1}')

# Plot the average correlation trend
plt.plot(periods, average_correlations_5, 'o-', color='black', label='Average')

plt.xlabel('Cycles')
plt.ylabel('PC (Correlation)')
plt.title('Correlation Trend of Features with Capacity B0005')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'Feature_6.xlsx'  
Feature_6 = pd.read_excel(file_path).iloc[1:168, 1:]
Capacity_6 = pd.DataFrame(capacity['B0006'][1]).iloc[1:168, :]

# Create an empty list to store the correlation trend between each feature and capacity over cycles
correlation_trend6 = []
periods = range(4, 171, 2)

# Loop to calculate the correlation between each feature and capacity over different cycles
for col, feature_data in Feature_6.items():
    correlations6 = []  # Store correlation values for each cycle

    for seg in periods:
        # Slice data based on the cycle period
        feature_slice = feature_data.iloc[:seg]
        capacity_slice = Capacity_6.iloc[:seg, 0]  # Capacity has only one column

        # Compute correlation
        correlation = feature_slice.corr(capacity_slice)

        # Add result to the list
        correlations6.append(abs(correlation))

    # Append the correlation trend of each feature with capacity to the overall list
    correlation_trend6.append(correlations6)

# Convert to a NumPy array for further processing
correlation_trend6 = np.array(correlation_trend6)

# Compute the average correlation for each cycle
average_correlations6 = np.mean(correlation_trend6, axis=0)

# Plot the correlation trend of each feature with capacity over cycles
plt.figure(figsize=(5, 3))
for i in range(correlation_trend6.shape[0]):
    plt.plot(periods, correlation_trend6[i], label=f'Feature {i+1}')

# Plot the average correlation trend
plt.plot(periods, average_correlations6, 'o-', color='black', label='Average')

plt.xlabel('Cycles')
plt.ylabel('PC (Correlation)')
plt.title('Correlation Trend of Features with Capacity B0006')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'E:\Feature_7.xlsx'  
Feature_7 = pd.read_excel(file_path).iloc[1:168, 1:]
Capacity_7 = pd.DataFrame(capacity['B0007'][1]).iloc[1:168, :]

# Create an empty list to store the correlation trend between each feature and capacity over cycles
correlation_trend = []
periods = range(4, 41, 2)

# Loop to calculate the correlation between each feature and capacity over different cycles
for col, feature_data in Feature_7.items():
    correlations = []  # Store correlation values for each cycle

    for seg in periods:
        # Slice data based on the cycle period
        feature_slice = feature_data.iloc[:seg]
        capacity_slice = Capacity_7.iloc[:seg, 0]  # Capacity has only one column

        # Compute correlation
        correlation = feature_slice.corr(capacity_slice)

        # Add result to the list
        correlations.append(abs(correlation))

    # Append the correlation trend of each feature with capacity to the overall list
    correlation_trend.append(correlations)

# Convert to a NumPy array for further processing
correlation_trend = np.array(correlation_trend)

# Compute the average correlation for each cycle
average_correlations = np.mean(correlation_trend, axis=0)

# Plot the correlation trend of each feature with capacity over cycles
plt.figure(figsize=(10, 6))
for i in range(correlation_trend.shape[0]):
    plt.plot(periods, correlation_trend[i], label=f'Feature {i+1}')

# Plot the average correlation trend
plt.plot(periods, average_correlations, 'o-', color='black', label='Average')

plt.xlabel('Cycles')
plt.ylabel('PC (Correlation)')
plt.title('Correlation Trend of Features with Capacity B0007')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Feature_5 = pd.read_excel(r'Feature_5.xlsx').iloc[1:168, 1:]
Feature_6 = pd.read_excel(r'Feature_6.xlsx').iloc[1:168, 1:]
Feature_7 = pd.read_excel(r'Feature_7.xlsx').iloc[1:168, 1:]

# Compute the minimum and maximum values for each column in Feature_5
min_vals_5 = Feature_5.min()
max_vals_5 = Feature_5.max()
# Perform min-max normalization on each column
normalized_Feature_5 = (Feature_5 - min_vals_5) / (max_vals_5 - min_vals_5)

# Compute the minimum and maximum values for each column in Feature_6
min_vals_6 = Feature_6.min()
max_vals_6 = Feature_6.max()
# Perform min-max normalization on each column
normalized_Feature_6 = (Feature_6 - min_vals_6) / (max_vals_6 - min_vals_6)

# Compute the minimum and maximum values for each column in Feature_7
min_vals_7 = Feature_7.min()
max_vals_7 = Feature_7.max()
# Perform min-max normalization on each column
normalized_Feature_7 = (Feature_7 - min_vals_7) / (max_vals_7 - min_vals_7)

# Create empty lists to store the Wasserstein distance trends for each feature
wasserstein_trend_56 = []  # Distance between Feature_5 and Feature_6
wasserstein_trend_57 = []  # Distance between Feature_5 and Feature_7
periods = range(4, 41, 2)

# Loop to calculate the Wasserstein distance trend for each feature over cycles
for col in range(5):
    wasserstein_values_56 = []  # Store Wasserstein distance between Feature_5 and Feature_6 for each feature
    wasserstein_values_57 = []  # Store Wasserstein distance between Feature_5 and Feature_7 for each feature

    for seg in periods:
        # Slice data based on the cycle period
        feature_data_5 = normalized_Feature_5.iloc[:seg, col]
        feature_data_6 = normalized_Feature_6.iloc[:seg, col]
        feature_data_7 = normalized_Feature_7.iloc[:seg, col]

        # Compute Wasserstein distance between features
        wasserstein_56 = wasserstein_distance(feature_data_5, feature_data_6)
        wasserstein_57 = wasserstein_distance(feature_data_5, feature_data_7)

        # Store 1 - Wasserstein distance to represent similarity
        wasserstein_values_56.append(1 - wasserstein_56)
        wasserstein_values_57.append(1 - wasserstein_57)

    # Append the Wasserstein distance trend for each feature
    wasserstein_trend_56.append(wasserstein_values_56)
    wasserstein_trend_57.append(wasserstein_values_57)

# Convert to NumPy arrays for further processing
wasserstein_trend_56 = np.array(wasserstein_trend_56)
wasserstein_trend_57 = np.array(wasserstein_trend_57)

# Compute the average Wasserstein distance for each cycle
average_wasserstein_56 = np.mean(wasserstein_trend_56, axis=0)
average_wasserstein_57 = np.mean(wasserstein_trend_57, axis=0)

# Plot the Wasserstein distance trend of features over cycles
plt.figure(figsize=(6, 4))
for i in range(5):
    #plt.plot(periods, wasserstein_trend_56[i], '-o', label=f'Feature {i+1} (5 vs 6)')
    plt.plot(periods, wasserstein_trend_57[i], '--o', label=f'Feature {i+1} (5 vs 7)')

# Plot the average Wasserstein distance trend
#plt.plot(periods, average_wasserstein_56, '-o', color='black', label='Average (5 vs 6)')
plt.plot(periods, average_wasserstein_57, '--o', color='black', label='Average (5 vs 7)')

plt.xlabel('Cycles')
plt.ylabel('TC')
plt.title('1-Wasserstein Distance between Features')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


wasserstein_trend_57=np.array(wasserstein_trend_57)
correlation_trend=np.array(correlation_trend)
correlation_trend_5=np.array(correlation_trend_5)
df_wasserstein = pd.DataFrame(wasserstein_trend_57.T, columns=[f'Feature_{i+1}' for i in range(5)])
df_correlation = pd.DataFrame(correlation_trend.T, columns=[f'Feature_{i+1}' for i in range(5)])
df_correlation_5 = pd.DataFrame(correlation_trend_5.T, columns=[f'Feature_{i+1}' for i in range(5)])

file_path = r'NASA.xlsx'
with pd.ExcelWriter(file_path) as writer:
    df_wasserstein.to_excel(writer, sheet_name='TC_Trend', index=False)
    df_correlation.to_excel(writer, sheet_name='PC_Trend', index=False)
    df_correlation_5.to_excel(writer, sheet_name='PC_Trend_5', index=False)


# In[11]:


# NASA PC TC Trend
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'NASA.xlsx'  
df_size = pd.read_excel(file_path, sheet_name='TC_Trend')  
df_color = pd.read_excel(file_path, sheet_name='PC_Trend')  
df_color5 = pd.read_excel(file_path, sheet_name='PC_Trend_5')  

mean_wasserstein = df_size.mean(axis=1)
mean_correlation = df_color.mean(axis=1)

fig, ax1 = plt.subplots(figsize=(4, 2), dpi=600)

color = '#599CB4'  
ax1.set_xlabel('Periods')
ax1.set_ylabel('Average Wasserstein Trend', color=color)
ax1.plot(mean_wasserstein, marker='o', markersize=10, linestyle='-', linewidth=5, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.8, 1)  

ax2 = ax1.twinx()  
color = '#C25759'  
ax2.set_ylabel('Average Correlation Trend', color=color)
ax2.plot(mean_correlation, marker='o', markersize=10, linestyle='-', linewidth=5, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1)  
plt.grid(False)
plt.show()


# In[6]:


#LSTM 
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
from sklearn.decomposition import PCA
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = r'Feature_6.xlsx'  
Feature_5 = pd.read_excel(file_path).iloc[1:168,1:]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_5 = scaler.fit_transform(pd.DataFrame(capacity['B0005'][1]).iloc[1:168,:])

n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'4(t-1)']
data_y5=processedCapacity_5.loc[:,'0']
data_y5=data_y5.values.reshape(-1,1)

train_X5=data_x5.values[:168]
train_y5=data_y5[:168]
train_X5 = train_X5.reshape((train_X5.shape[0], n_steps_in, 5))


# In[7]:


# Train source model 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from tensorflow.keras.models import save_model, load_model
import os
import time

np.random.seed(42)
tf.random.set_seed(42)

start_time=time.time()
model = Sequential()
model.add(LSTM(96, return_sequences=True, input_shape=(train_X5.shape[1], train_X5.shape[2]))) 
model.add(LSTM(64, return_sequences=False))

model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

history = model.fit(train_X5, train_y5, epochs=30, batch_size=64,verbose=2, shuffle=False)
end_time = time.time()
elapsed_time = end_time - start_time


# In[ ]:


# Souce domain is 5，target domain is 6. Automatically observe the situation of different fine-tuning periods
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from keras.models import Model
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import save_model, load_model
import os
import time


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

file_path_5 = r'Feature_5.xlsx'  
Feature_5 = pd.read_excel(file_path_5).iloc[1:168,1:]
file_path_6 = r'Feature_6.xlsx'  
Feature_6 = pd.read_excel(file_path_6).iloc[1:168,1:]

capacity_5=pd.DataFrame(capacity['B0005'][1]).iloc[1:168,:]
capacity_6=pd.DataFrame(capacity['B0006'][1]).iloc[1:168,:]
n_steps_in=3
n_steps_out = 1
np.random.seed(42)
tf.random.set_seed(42)

prediction_capability=[]
for train_samples in range(2,41,2):

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledFeature_6 = scaler.fit_transform(Feature_6)
    
    scaledFeature_6 = pd.DataFrame(data=scaledFeature_6)
  
    n_steps_in =3 
    n_steps_out=1
    processedFeature_6 = time_series_to_supervised(scaledFeature_6,n_steps_in,n_steps_out)
 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledCapacity_6 = scaler.fit_transform(pd.DataFrame(capacity['B0006'][1]).iloc[1:168,:])
  
    n_steps_in =3 
    n_steps_out=1
    processedCapacity_6 = time_series_to_supervised(scaledCapacity_6,n_steps_in,n_steps_out)
   
    data_x6 = processedFeature_6.loc[:,'0(t-3)':'4(t-1)']
    data_y6=processedCapacity_6.loc[:,'0']
    data_y6=data_y6.values.reshape(-1,1)
  
    train_X6=data_x6.values[:train_samples]
    test_X6=data_x6.values[train_samples:]
    train_y6=data_y6[:train_samples]
    test_y6=data_y6[train_samples:]
    train_X6 = train_X6.reshape((train_X6.shape[0], n_steps_in, 5))
    test_X6 = test_X6.reshape((test_X6.shape[0], n_steps_in, 5))
    
    for layer in model.layers[:2]:  
        layer.trainable = False

    input_layer = Input(shape=(train_X6.shape[1], train_X6.shape[2]))
    lstm_output_1 = model.layers[0](input_layer)  
    lstm_output_2 = model.layers[1](lstm_output_1) 
    new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
    #new_dense_1 = model.layers[2](lstm_output_2)
    new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)
    #new_output_layer = model.layers[3](new_dense_1)

    transfer_model = Model(inputs=input_layer, outputs=new_output_layer)
    transfer_model.compile(loss='mse', optimizer='adam')
    transfer_model.fit(train_X6, train_y6, epochs=50, batch_size=64, verbose=2, shuffle=False)

    yhat6t= transfer_model.predict(test_X6)
    test_y6=test_y6.reshape(-1,1) 
   
    inv_forecast_y6t = scaler.inverse_transform(yhat6t)
    inv_test_y6t = scaler.inverse_transform(test_y6)
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true))
    mape_6t = mape(inv_test_y6t, inv_forecast_y6t)
    print('Test MAPE: %.3f' % mape_6t)
    prediction_capability.append(1-mape_6t)
prediction_capability


# In[ ]:


# Souce domain is 5，target domain is 7. Automatically observe the situation of different fine-tuning periods
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from keras.models import Model
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import save_model, load_model
import os
import time


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

file_path_5 = r'Feature_5.xlsx'  
Feature_5 = pd.read_excel(file_path_5).iloc[1:168,1:]
file_path_7 = r'Feature_7.xlsx'  
Feature_7 = pd.read_excel(file_path_7).iloc[1:168,1:]

capacity_5=pd.DataFrame(capacity['B0005'][1]).iloc[1:168,:]
capacity_7=pd.DataFrame(capacity['B0007'][1]).iloc[1:168,:]
n_steps_in=3
n_steps_out = 1

np.random.seed(42)
tf.random.set_seed(42)

prediction_capability7=[]
for train_samples in range(2,41,2):

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledFeature_7 = scaler.fit_transform(Feature_7)
    
    scaledFeature_7 = pd.DataFrame(data=scaledFeature_7)
   
    n_steps_in =3 
    n_steps_out=1
    processedFeature_7 = time_series_to_supervised(scaledFeature_7,n_steps_in,n_steps_out)
  
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledCapacity_7 = scaler.fit_transform(pd.DataFrame(capacity['B0007'][1]).iloc[1:168,:])
    
    n_steps_in =3 
    n_steps_out=1
    processedCapacity_7 = time_series_to_supervised(scaledCapacity_7,n_steps_in,n_steps_out)
   
    data_x7 = processedFeature_7.loc[:,'0(t-3)':'4(t-1)']
    data_y7=processedCapacity_7.loc[:,'0']
    data_y7=data_y7.values.reshape(-1,1)

    train_X7=data_x7.values[:train_samples]
    test_X7=data_x7.values[train_samples:]
    train_y7=data_y7[:train_samples]
    test_y7=data_y7[train_samples:]
    train_X7 = train_X7.reshape((train_X7.shape[0], n_steps_in, 5))
    test_X7 = test_X7.reshape((test_X7.shape[0], n_steps_in, 5))
    
    for layer in model.layers[:2]: 
        layer.trainable = False

    input_layer = Input(shape=(train_X7.shape[1], train_X7.shape[2]))
    lstm_output_1 = model.layers[0](input_layer)  
    lstm_output_2 = model.layers[1](lstm_output_1)  
    new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
    #new_dense_1 = model.layers[2](lstm_output_2)
    new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)
    #new_output_layer = model.layers[3](new_dense_1)

    transfer_model = Model(inputs=input_layer, outputs=new_output_layer)
    transfer_model.compile(loss='mse', optimizer='adam')
    transfer_model.fit(train_X7, train_y7, epochs=50, batch_size=64, verbose=2, shuffle=False)

    yhat7t= transfer_model.predict(test_X7)
    test_y7=test_y7.reshape(-1,1) 
   
    inv_forecast_y7t = scaler.inverse_transform(yhat7t)
    inv_test_y7t = scaler.inverse_transform(test_y7)
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true))
    mape_7t = mape(inv_test_y7t, inv_forecast_y7t)
    print('Test MAPE: %.3f' % mape_7t)
    prediction_capability7.append(1-mape_7t)
prediction_capability7


# In[ ]:


# NASA Acc
import pandas as pd

prediction_df = pd.DataFrame(prediction_capability7, columns=["5 to 7"])

file_path_output = "NASA_Acc.xlsx"

prediction_df.to_excel(file_path_output, index=False)


# In[ ]:


# Acc plot
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'NASA_Acc.xlsx' 
df = pd.read_excel(file_path)

column_data = df['5 to 7']

plt.figure(figsize=(5, 2), dpi=600)

for i in range(len(column_data)):
    if i == 3:  
        plt.bar(i, column_data[i], width=0.8, color='#898988')
    else:
        plt.bar(i, column_data[i], width=0.8, color='#d7d7d7')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.grid(False)
plt.ylim(0.985,0.991)
plt.show()


# In[ ]:


#True-Pre Curve
#LSTM 
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
from sklearn.decomposition import PCA
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = r'Feature_6.xlsx'  
Feature_5 = pd.read_excel(file_path).iloc[1:168,1:]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_5 = scaler.fit_transform(pd.DataFrame(capacity['B0005'][1]).iloc[1:168,:])

n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'4(t-1)']
data_y5=processedCapacity_5.loc[:,'0']
data_y5=data_y5.values.reshape(-1,1)

train_X5=data_x5.values[:168]
train_y5=data_y5[:168]
train_X5 = train_X5.reshape((train_X5.shape[0], n_steps_in, 5))


# In[ ]:


# Train source model 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from tensorflow.keras.models import save_model, load_model
import os
import time

np.random.seed(42)
tf.random.set_seed(42)

start_time=time.time()
model = Sequential()
model.add(LSTM(96, return_sequences=True, input_shape=(train_X5.shape[1], train_X5.shape[2]))) 
model.add(LSTM(64, return_sequences=False))

model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

history = model.fit(train_X5, train_y5, epochs=30, batch_size=64,verbose=2, shuffle=False)
end_time = time.time()
elapsed_time = end_time - start_time


# In[11]:


# B7 Validation no finetune
import matplotlib
train_samples=10

selected_columns = ['Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
file_path_6 = r'Feature_7.xlsx'  
Feature_6 = pd.read_excel(file_path_6).iloc[1:168,1:]



#Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[:800]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_6 = scaler.fit_transform(Feature_6)
scaledFeature_6 = pd.DataFrame(data=scaledFeature_6)

n_steps_in =3 
n_steps_out=1
processedFeature_6 = time_series_to_supervised(scaledFeature_6,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
#cap_6=(pd.read_excel(file_path_6).iloc[:800,-1]).values.reshape(-1, 1)
scaledCapacity_6 = scaler.fit_transform(pd.DataFrame(capacity['B0007'][1]).iloc[1:168,:])

n_steps_in =3 
n_steps_out=1
processedCapacity_6 = time_series_to_supervised(scaledCapacity_6,n_steps_in,n_steps_out)
data_x6 = processedFeature_6.loc[:,'0(t-3)':'4(t-1)']
data_y6=processedCapacity_6.loc[:,'0']
data_y6=data_y6.values.reshape(-1,1)
train_X6=data_x6.values[:train_samples]
test_X6=data_x6.values[train_samples:]
train_y6=data_y6[:train_samples]
test_y6=data_y6[train_samples:]
train_X6 = train_X6.reshape((train_X6.shape[0], n_steps_in, 5))
test_X6 = test_X6.reshape((test_X6.shape[0], n_steps_in, 5))

yhat6 = model.predict(test_X6)
test_y6=test_y6.reshape(-1,1)
inv_forecast_y6 = scaler.inverse_transform(yhat6)
inv_test_y6 = scaler.inverse_transform(test_y6)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

mape_6 = mape(inv_test_y6, inv_forecast_y6)
print('Test MAPE: %.3f' % mape_6)

plt.figure(figsize=(8,6))
plt.plot(pd.DataFrame(capacity['B0007'][1]).iloc[1:168,:], label='True')
x_range = range(train_samples, train_samples+ len(inv_forecast_y6))
plt.plot(x_range,inv_forecast_y6,marker='.',label='LSTM',linestyle=None,markersize=5)

plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[12]:


# B7 Validation with finetune
start_time = time.time()

from keras.models import Model
from tensorflow.keras.layers import Input

for layer in model.layers[:2]:  
    layer.trainable = False

input_layer = Input(shape=(train_X6.shape[1], train_X6.shape[2]))
lstm_output_1 = model.layers[0](input_layer)  
lstm_output_2 = model.layers[1](lstm_output_1)  
new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)

transfer_model = Model(inputs=input_layer, outputs=new_output_layer)
transfer_model.compile(loss='mse', optimizer='adam')
transfer_model.fit(train_X6, train_y6, epochs=50, batch_size=64, verbose=2, shuffle=False)

yhat6t= transfer_model.predict(test_X6)
test_y6=test_y6.reshape(-1,1) 
inv_forecast_y6t = scaler.inverse_transform(yhat6t)
inv_test_y6t = scaler.inverse_transform(test_y6)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
mape_6t = mape(inv_test_y6t, inv_forecast_y6t)
print('Test MAPE: %.3f' % mape_6t)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Model running time：{elapsed_time} Seconds")

plt.figure(figsize=(8,6))
plt.plot(pd.DataFrame(capacity['B0007'][1]).iloc[1:168,:], label='True')
x_range = range(train_samples, train_samples+len(inv_forecast_y6t))
plt.plot(x_range,inv_forecast_y6t,marker='.',label='LSTM+Fine-tune',linestyle=None,markersize=5)
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame(capacity['B0007'][1]).iloc[1:168,:]
initial_capacity = data.iloc[0, -1] 
threshold_capacity = 0.95 * initial_capacity  

plt.figure(figsize=(4, 3),dpi=600)
plt.plot(data.iloc[:800, -1], label='Target_True',linewidth=5,color=plt.cm.Blues(0.8))
plt.plot(capacity['B0005'][1], label='Source',linewidth=5,color=plt.cm.Reds(0.8))

x_range = range(train_samples, train_samples + len(inv_forecast_y6t))
plt.plot(x_range, inv_forecast_y6t,label='Target_Pre', linestyle=None,linewidth=5,color=plt.cm.Greens(0.8))

def find_intersection(x_vals, y_vals, threshold):
    for i in range(len(y_vals) - 1):
        if (y_vals[i] >= threshold and y_vals[i + 1] < threshold) or (y_vals[i] <= threshold and y_vals[i + 1] > threshold):
            return x_vals[i]  
    return None

true_x_intersection = train_samples+find_intersection(range(len(inv_test_y6t)), inv_test_y6t, threshold_capacity)
pred_x_intersection = find_intersection(x_range, inv_forecast_y6t.flatten(), threshold_capacity)

plt.ylabel('Capacity(Ah)', fontsize=12)
plt.xlabel('Cycle', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('NASA-2.7V#7')
plt.ylim(initial_capacity*0.7,initial_capacity*1.05)
plt.show()

print(f"True EOL: {true_x_intersection}")
print(f"Pre EOL: {pred_x_intersection}")


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# HUST voltage-SOC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle
import matplotlib.cm as cm

with open(r'HUST\Data\8-2.pkl', 'rb') as f:
    data = pickle.load(f)

num_cycles = len(data['8-2']['data'])
cmap = cm.get_cmap('Blues', num_cycles)  
colors = cmap(np.linspace(0.3, 1, num_cycles))  

plt.figure(figsize=(4, 3),dpi=600)

for cycle in range(1, num_cycles + 1):
    df = data['8-2']['data'][cycle]

    V1 = df[df['Status'] == 'Constant current charge']['Voltage (V)'].values
    t1 = df[df['Status'] == 'Constant current charge']['Capacity (mAh)'].values
 
    V2 = df[df['Status'] == 'Constant current-constant voltage charge']['Voltage (V)'].values
    t2 = df[df['Status'] == 'Constant current-constant voltage charge']['Capacity (mAh)'].values
   
    time = np.concatenate([t1, t2]) - t1[0]  
    voltage = np.concatenate([V1, V2])

    plt.plot(time, voltage, linewidth=1, alpha=0.6,color=colors[cycle - 1])

plt.title('Voltage vs Capacity for All Cycles')
plt.xlabel('Charging Capacity (mAh)')
plt.ylabel('Charging Voltage (V)')
plt.ylim(2,4.25)
plt.show()


# In[2]:


#HUST capacity-cycle
import pandas as pd
import matplotlib.pyplot as plt
import os

file1 = r'Features_HUST_42_filtered.xlsx'
file2 = r'Features_HUST_82_filtered.xlsx'

data1 = pd.read_excel(file1, header=None)  
capacity1 = data1.iloc[1:,9]  

data2 = pd.read_excel(file2, header=None)  
capacity2 = data2.iloc[1:, 9] 

plt.figure(figsize=(4, 3), dpi=600)
plt.plot(capacity1, label='File 2', marker='o', markersize=3, color=plt.cm.Reds(0.8))
plt.plot(capacity2, label='File 1', marker='o', markersize=3, color=plt.cm.Blues(0.8))

plt.title('HUST', fontsize=10)
plt.xlabel('Cycle Index', fontsize=10)
plt.ylabel('Capacity (Ah)', fontsize=10)

plt.tight_layout()
plt.show()


# In[ ]:


# Feature extraction +savgol_filter 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter

f = open(r'HUST\8-2.pkl','rb')#change file name to extract different battery features
data = pickle.load(f)

results = []

for cycle in range(len(data['8-2']['data'])):
    df = data['8-2']['data'][cycle+1]
    
    V1 = df[df['Status'] == 'Constant current charge']['Voltage (V)'][10:]
    t1 = df[df['Status'] == 'Constant current charge']['Time (s)'][10:]
    
    V2 = df[df['Status'] == 'Constant current-constant voltage charge']['Voltage (V)']

    voltage_diff = V2.diff().fillna(0)
    
    threshold = 0.0005
    increasing_indices = voltage_diff[voltage_diff > threshold].index

    increasing_voltage = V2[increasing_indices]
    increasing_time = df[df['Status'] == 'Constant current-constant voltage charge']['Time (s)'][increasing_indices]
    
    Gra_1 = np.diff(V1) / np.diff(t1)
    Vg1 = np.mean(Gra_1)
    
    Gra_2 = np.diff(increasing_voltage) / np.diff(increasing_time)
    Vg2 = np.mean(Gra_2)
    
    RL1 = (df[df['Status'] == 'Constant current charge']['Voltage (V)'].iloc[-1] - 
           df[df['Status'] == 'Constant current charge']['Voltage (V)'].iloc[0]) / 5.5
    RL2 = (V2.iloc[-1] - V2.iloc[0]) / 1.1
    RO = (V1.iloc[-1] - V2.iloc[0]) / (5.5 - 1.1)  
    
    Q1 = df[df['Status'] == 'Constant current charge']['Capacity (mAh)'].iloc[-1]
    Q2 = df[df['Status'] == 'Constant current-constant voltage charge']['Capacity (mAh)'][increasing_indices].iloc[-1]
    
    tVD2 = increasing_time.iloc[-1]
    cap = df[df['Status'] == 'Constant current discharge_0']['Capacity (mAh)'].iloc[0]

    cycle_features = {
        'Cycle': cycle+1,
        'Vg1': Vg1,
        'Vg2': Vg2,
        'Q1': Q1,
        'Q2': Q2,
        'RL1': RL1,
        'RL2': RL2,
        'RO': RO,
        'tVD2': tVD2,
        'Capacity': cap
    }

    results.append(cycle_features)

results_df = pd.DataFrame(results)

window_length = 11  
polyorder = 2  

for col in results_df.columns:
    if col != 'Cycle' and col != 'Capacity':
        results_df[col] = savgol_filter(results_df[col], window_length, polyorder)

print(results_df)

results_df.to_excel('Features_HUST_82_filtered.xlsx', index=False)

print("eature extraction and filtering are complete and saved")


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'Features_HUST_82_filtered.xlsx'
df = pd.read_excel(file_path).iloc[5:,]

columns_to_analyze = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']

def calculate_correlations(df, cycles, target='Capacity'):
    cycle_data = df[df['Cycle'] <= cycles]
    correlations = cycle_data[columns_to_analyze + [target]].corr()[target].drop(target)
    return abs(correlations)

correlations = {}
for cycle in range(20, len(df)+1,20):
    correlations[cycle] = calculate_correlations(df, cycle)

average_correlations = np.mean([list(correlations[cycle].values) for cycle in range(20, len(df)+1,20)], axis=1)

plt.figure(figsize=(10, 6),dpi=600)

for feature in columns_to_analyze:
    corr_values = [correlations[cycle][feature] for cycle in range(20, len(df)+1,20)]
    plt.plot(range(20, len(df)+1,20), corr_values, marker='o', linestyle='-', label=feature)

plt.plot(range(20, len(df)+1,20), average_correlations, marker='o', linestyle='-', color='black', label='Average')

plt.title('Correlations with Capacity 8-2')
plt.xlabel('Cycles')
plt.ylabel('Correlation')
plt.xticks(range(100, len(df) + 1, 100))
plt.legend()
plt.tight_layout()
plt.show()


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'Features_HUST_42_filtered.xlsx'
df = pd.read_excel(file_path).iloc[5:,]

columns_to_analyze = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']

def calculate_correlations(df, cycles, target='Capacity'):
    cycle_data = df[df['Cycle'] <= cycles]
    correlations = cycle_data[columns_to_analyze + [target]].corr()[target].drop(target)
    return abs(correlations)

correlations_42 = {}
for cycle in range(20, len(df)+1,20):
    correlations_42[cycle] = calculate_correlations(df, cycle)

average_correlations = np.mean([list(correlations_42[cycle].values) for cycle in range(20, len(df)+1,20)], axis=1)

plt.figure(figsize=(10, 6),dpi=600)

for feature in columns_to_analyze:
    corr_values = [correlations_42[cycle][feature] for cycle in range(20, len(df)+1,20)]
    plt.plot(range(20, len(df)+1,20), corr_values, marker='o', linestyle='-', label=feature)

plt.plot(range(20, len(df)+1,20), average_correlations, marker='o', linestyle='-', color='black', label='Average')

plt.title('Correlations with Capacity 4-2')
plt.xlabel('Cycles')
plt.ylabel('Correlation')
plt.xticks(range(100, len(df) + 1, 100))
plt.legend()
plt.tight_layout()
plt.show()


# In[7]:


import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

file_path_1 = r'Features_HUST_42_filtered.xlsx'
file_path_2 = r'Features_HUST_82_filtered.xlsx'

df1 = pd.read_excel(file_path_1).iloc[5:1700, :]
df2 = pd.read_excel(file_path_2).iloc[5:1700, :]

common_columns = list(set(df1.columns) & set(df2.columns))
columns_to_analyze=[ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

df1[columns_to_analyze] = scaler1.fit_transform(df1[columns_to_analyze])
df2[columns_to_analyze] = scaler2.fit_transform(df2[columns_to_analyze])

def calculate_wasserstein_distance(df1, df2, cycles, columns):
    distances = {}
    for col in columns:
        data1 = df1.iloc[:cycles][col].dropna()
        data2 = df2.iloc[:cycles][col].dropna()
        if not data1.empty and not data2.empty:
            distance = wasserstein_distance(data1, data2)
            distances[col] = 1 - distance
    return distances

max_cycles = min(df1['Cycle'].max(), df2['Cycle'].max())
cycle_ranges = range(20, max_cycles + 1, 20)

wasserstein_distances = {col: [] for col in columns_to_analyze}
average_distances = []

for cycle_range in cycle_ranges:
    distances = calculate_wasserstein_distance(df1, df2, cycle_range, columns_to_analyze)
    for col, distance in distances.items():
        wasserstein_distances[col].append(distance)

    if distances:
        average_distance = np.mean(list(distances.values()))
        average_distances.append(average_distance)
    else:
        average_distances.append(0)

plt.figure(figsize=(10, 6),dpi=600)

for col, distances in wasserstein_distances.items():
    plt.plot(cycle_ranges, distances, marker='o', linestyle='-', label=col)

plt.plot(cycle_ranges, average_distances, marker='o', linestyle='--', color='black', label='Average')
plt.ylim()
plt.title('1-Wasserstein Distance between Features')
plt.xlabel('Cycles')
plt.ylabel('1-Wasserstein Distance')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


#HUST PC TC excel
import pandas as pd
import numpy as np

df_wasserstein = pd.DataFrame(wasserstein_distances)
correlations_df = pd.DataFrame(correlations)
correlations_df_42 = pd.DataFrame(correlations_42)

correlations_df = correlations_df.T
df_correlation = pd.DataFrame(correlations_df)
correlations_df_42 = correlations_df_42.T
df_correlation_42 = pd.DataFrame(correlations_df_42)

file_path = 'HUST.xlsx'
with pd.ExcelWriter(file_path) as writer:
    df_wasserstein.to_excel(writer, sheet_name='TC_Trend', index=False)
    df_correlation.to_excel(writer, sheet_name='PC_Trend', index=False)
    df_correlation_42.to_excel(writer, sheet_name='PC_Trend_42', index=False)

print(f"Data has been successfully written {file_path}")


# In[6]:


#PC TC Trend
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'HUST.xlsx'  
df_TC = pd.read_excel(file_path, sheet_name='TC_Trend')  
df_PC = pd.read_excel(file_path, sheet_name='PC_Trend')  

mean_wasserstein = df_TC.mean(axis=1)
mean_correlation = df_PC.mean(axis=1)

fig, ax1 = plt.subplots(figsize=(4, 2), dpi=600)

color = '#599CB4'  
ax1.set_xlabel('Periods')
ax1.set_ylabel('Average Wasserstein Trend', color=color)
ax1.plot(mean_wasserstein[:20], marker='o', markersize=7,linestyle='-', linewidth=4,color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.7, 1.0)  

ax2 = ax1.twinx() 
color = '#C25759'  
ax2.set_ylabel('Average Correlation Trend', color=color)
ax2.plot(mean_correlation[:20], marker='o', markersize=10,linestyle='-', linewidth=5,color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.2, 1)  
plt.show()


# In[ ]:





# In[2]:


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

file_path_5 = r'Features_HUST_42_filtered.xlsx'
selected_columns = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[5:1706]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_5).iloc[5:1706,-1]).values.reshape(-1, 1)
scaledCapacity_5 = scaler.fit_transform(cap)
n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'6(t-1)']
data_y5=processedCapacity_5.loc[:,'0']
train_X5=data_x5.values[:1706]
train_y5=data_y5[:1706]
train_X5 = train_X5.reshape((train_X5.shape[0], n_steps_in, 7))


# In[3]:


#Train source model
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


# Automatically look at different fine-tuning cycles
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

file_path_5 = r'Features_HUST_42_filtered.xlsx'  
selected_columns = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[5:1706]
file_path_6 = r'Features_HUST_82_filtered.xlsx'  
Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[5:2283]

capacity_5=pd.read_excel(file_path_5).iloc[5:1706,-1]
capacity_6=pd.read_excel(file_path_6).iloc[5:2283,-1]
cap_6=(capacity_6).values.reshape(-1, 1)
n_steps_in=3
n_steps_out = 1

np.random.seed(42)
tf.random.set_seed(42)

prediction_capability=[]
for train_samples in range(20,401,20):

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledFeature_6 = scaler.fit_transform(Feature_6)
   
    scaledFeature_6 = pd.DataFrame(data=scaledFeature_6)
    
    n_steps_in =3 
    n_steps_out=1
    processedFeature_6 = time_series_to_supervised(scaledFeature_6,n_steps_in,n_steps_out)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledCapacity_6 = scaler.fit_transform(cap_6)
   
    n_steps_in =3 
    n_steps_out=1
    processedCapacity_6 = time_series_to_supervised(scaledCapacity_6,n_steps_in,n_steps_out)
   
    data_x6 = processedFeature_6.loc[:,'0(t-3)':'6(t-1)']
    data_y6=processedCapacity_6.loc[:,'0']
    data_y6=data_y6.values.reshape(-1,1)
   
    train_X6=data_x6.values[:train_samples]
    test_X6=data_x6.values[train_samples:]
    train_y6=data_y6[:train_samples]
    test_y6=data_y6[train_samples:]
    train_X6 = train_X6.reshape((train_X6.shape[0], n_steps_in, 7))
    test_X6 = test_X6.reshape((test_X6.shape[0], n_steps_in, 7))
    
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
    prediction_capability.append(1-mape_6t)
prediction_capability


# In[ ]:


# HUST Acc
import pandas as pd

prediction_df = pd.DataFrame(prediction_capability, columns=["42 to 82"])

file_path_output = "HUST_Acc.xlsx"

prediction_df.to_excel(file_path_output, index=False)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

file_path = 'HUST_Acc.xlsx'  
df = pd.read_excel(file_path)

column_data = df['42 to 82']
column_data = df['42 to 82']

plt.figure(figsize=(5, 2), dpi=600)

for i in range(len(column_data)):
    if i == 2:  
        plt.bar(i, column_data[i], width=0.8, color='#898988')
    else:
        plt.bar(i, column_data[i], width=0.8, color='#d7d7d7')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.grid(False)
plt.ylim(0.97,0.998)
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

file_path_5 = r'Features_HUST_42_filtered.xlsx'
selected_columns = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[5:1706]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_5).iloc[5:1706,-1]).values.reshape(-1, 1)
scaledCapacity_5 = scaler.fit_transform(cap)
n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'6(t-1)']
data_y5=processedCapacity_5.loc[:,'0']
train_X5=data_x5.values[:1706]
train_y5=data_y5[:1706]
train_X5 = train_X5.reshape((train_X5.shape[0], n_steps_in, 7))


# In[ ]:


#Train source model
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


# In[4]:


# T45 Validation no finetune
import matplotlib
train_samples=60

selected_columns = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']
file_path_6 = r'Features_HUST_82_filtered.xlsx'  
Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[5:2283]
capacity_6=pd.read_excel(file_path_6).iloc[5:2283,-1]
cap_6=capacity_6.values.reshape(-1, 1)


#Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[:800]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_6 = scaler.fit_transform(Feature_6)
scaledFeature_6 = pd.DataFrame(data=scaledFeature_6)

n_steps_in =3 
n_steps_out=1
processedFeature_6 = time_series_to_supervised(scaledFeature_6,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
#cap_6=(pd.read_excel(file_path_6).iloc[:800,-1]).values.reshape(-1, 1)
scaledCapacity_6 = scaler.fit_transform(cap_6)

n_steps_in =3 
n_steps_out=1
processedCapacity_6 = time_series_to_supervised(scaledCapacity_6,n_steps_in,n_steps_out)
data_x6 = processedFeature_6.loc[:,'0(t-3)':'6(t-1)']
data_y6=processedCapacity_6.loc[:,'0']
data_y6=data_y6.values.reshape(-1,1)
train_X6=data_x6.values[:train_samples]
test_X6=data_x6.values[train_samples:]
train_y6=data_y6[:train_samples]
test_y6=data_y6[train_samples:]
train_X6 = train_X6.reshape((train_X6.shape[0], n_steps_in, 7))
test_X6 = test_X6.reshape((test_X6.shape[0], n_steps_in, 7))

yhat6 = model.predict(test_X6)
test_y6=test_y6.reshape(-1,1)
inv_forecast_y6 = scaler.inverse_transform(yhat6)
inv_test_y6 = scaler.inverse_transform(test_y6)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

mape_6 = mape(inv_test_y6, inv_forecast_y6)
print('Test MAPE: %.3f' % mape_6)

plt.figure(figsize=(8,6))
plt.plot(pd.read_excel(file_path_6).iloc[:2283,-1], label='True')
x_range = range(train_samples, train_samples+ len(inv_forecast_y6))
plt.plot(x_range,inv_forecast_y6,marker='.',label='LSTM',linestyle=None,markersize=5)

plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[5]:


# T45 Validation with finetune
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
plt.plot(pd.read_excel(file_path_6).iloc[:2283,-1], label='True')
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

data = pd.read_excel(file_path_6)
initial_capacity = data.iloc[0, -1]  
threshold_capacity = 0.95 * initial_capacity 

plt.figure(figsize=(4, 3),dpi=600)
plt.plot(data.iloc[:2283, -1], label='Target_True',linewidth=5,color=plt.cm.Blues(0.8))
plt.plot(pd.read_excel(file_path_5).iloc[5:1706,-1], label='Source',linewidth=5,color=plt.cm.Reds(0.8))


x_range = range(train_samples, train_samples + len(inv_forecast_y6t))
plt.plot(x_range, inv_forecast_y6t,label='Target_Pre', linestyle=None, linewidth=5,color=plt.cm.Greens(0.8))

def find_intersection(x_vals, y_vals, threshold):
    for i in range(len(y_vals) - 1):
        if (y_vals[i] >= threshold and y_vals[i + 1] < threshold) or (y_vals[i] <= threshold and y_vals[i + 1] > threshold):
            return x_vals[i]  
    return None

true_x_intersection = train_samples+find_intersection(range(len(inv_test_y6t)), inv_test_y6t, threshold_capacity)
pred_x_intersection = find_intersection(x_range, inv_forecast_y6t.flatten(), threshold_capacity)

plt.ylabel('Capacity(mAh)', fontsize=12)
plt.xlabel('Cycle', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('HUST-3C')
plt.show()

print(f"True EOL: {true_x_intersection}")
print(f"Pre EOL: {pred_x_intersection}")


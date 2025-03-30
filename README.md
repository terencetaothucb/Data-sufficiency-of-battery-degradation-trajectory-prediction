# Data-Sufficiency-of-Battery-Degradation-Trajectory-Prediction
This repository is for the data sufficiency theory for data-driven battery degradation prediction. Fundamentally, it answers the question: how much early cycle data is sufficient for robust and generalizable early predictions.
# 1.Set Up
## 1.1 Enviroments
- Python (Jupyter notebook)
## 1.2 Python requirements
- numpy=1.22.4
- tensorflow=2.10.0
- keras=2.10.0
- matplotlib=3.8.3
- scipy=1.7.3
- scikit-learn=1.3.0
- pandas=2.0.3
# 2. Datasets
The seven datasets encompass 6 materials, 7 working conditions(different multi-stage charging, different constant current–constant voltage (CCCV), and multi-stage discharging) and 7 transfer scenarios (different temperatures, discharging cut-off voltage, different charging rates, and discharging rate). Even under the same CCCV mode, the current and voltage vary across datasets.
| Dataset | Material       | Working Condition      | Transfer scenarios             |
|---------|----------------|------------------------|---------------------------------|
| THU     | NCM (811)      | 9 stage-Charging       | Temperature                     |
| NASA    | NCM (333)      | CCCV                   | Discharging Cut-off Voltage     |
| TJU     | NCA            | CCCV                   | Temperature                     |
| CALCE   | LCO            | CCCV                   | Charging Rate                   |
| MIT     | LFP            | 2 stage-Charging       | Charging Rate                   |
| XJTU    | NCM (523)      | CCCV                   | Charging Rate                   |
| HUST    | LFP            | 4 stage-Discharging    | Discharging Rate                |

We align the features of seven datasets with the physical meanings. Among them, CCCV datasets include NASA, TJU, CALCE and XJTU, from which 5 features can be extracted. Multi-stage datasets including THU, MIT and HUST, allowing for extracting a wider range of features, as both intra-step and inter-step features are available.

| Dataset | Vg               | Q                | RL               | RO               | VC VD / Other Variables                     |
|---------|------------------|------------------|------------------|------------------|---------------------------------------------|
|         | Polarization Speed | Charging Capacity | Lumped Resistance | Ohmic Resistance | Polarization                                |
| THU     | Vg1-Vg9          | Q1-Q9            | RL1-RL9          | RO1-RO8          | VC89, VD9, tVD9, ReVC, ReVD, tReVD          |
| NASA    | Vg               | Q                | RL               | -                | T_CCCV<br>T_CVCA                            |
| TJU     | Vg               | Q                | RL               | -                | T_CCCV<br>T_CVCA                            |
| CALCE   | Vg               | Q                | RL               | -                | T_CCCV<br>T_CVCA                            |
| MIT     | Vg1 Vg2          | -                | RL1 RL2          | RO               | VC21, VD2, tVD2, ReVC, ReVD, tReVD          |
| XJTU    | Vg               | Q                | RL               | -                | T_CCCV<br>T_CVCA                            |
| HUST    | Vg1 Vg2          | Q1 Q2            | RL1 RL2          | RO               | tVD2                                       |

The calculation and physical meanings of features are explained:
- Vg: Mean value of voltage gradient at charging process. The physical meaning is polarization speed.
- Q: Charging capacity value when the charging SOC meets the requirement. The physical meaning is charging ability in charging process.
- RL: Ratio of voltage and charging current during charging. The physical meaning is merged representation of ohmic, electrochemical, and concentration resistance.
- RO: Ratio of voltage change and current change at switching points between charging stages. The physical meaning is ohmic resistance from relaxation behaviours.
- VC/VD group, T_CCCV, T_CVCA: Voltage change in different stages or voltage drop within the own stage or the time needed to charge. The physical meaning is ohmic, electrochemical and concentration polarization.
# 3. Data Sufficiency 
Data sufficiency (DS) refers to the minimum amount of data required to achieve a high prediction accuracy, where further data collection does not lead to a significant improvement in accuracy or reduction in prediction error.
## 3.1 Observable data sufficiency(ODS)
ODS represents the cycle at which the prediction accuracy reaches its peak without further improvement. It identifies the point at which additional data no longer significantly enhances the model’s predictive performance, ensuring efficient use of available data while minimizing redundancy. 
The accuracy is defined as:

$$ Acc = 1 - MAPE $$

To identify the contribution of each additional data interval, the change in accuracy, $$\Delta Acc(j)$$, between successive intervals is calculated as:

$$\Delta Acc(j) = Acc(j) - Acc(j-1) $$

The ODS is determined as the cycle index $$j $$ where the accuracy continues to increase but subsequently stops improving, defined as:

$$ ODS = \arg\max_j [\Delta Acc(j) > 0, \Delta Acc(j+1) < 0] \quad $$
## 3.2 Theoretical data sufficiency (TDS)
We quantify the features' ability to predict capacity accurately and maintain performance under varying operating conditions in transfer learning scenarios. Thus, the prediction capability(PC) and transferable capability(TC) are defined. The combination of $$PC$$ and $$TC$$ forms the foundation for defining $$TDS$$. $$TDS$$ indicates the cycle at which the $$PC$$ and $$TC$$ are both high, marking the theoretical point of sufficient data. 

The combined metric integrates normalized TC and PC values while emphasizing their decline to highlight the current cycle's comprehensive value:

$$ f (PC,TC) = TC_{\text{norm}}(j) + PC_{\text{norm}}(j) + \Delta TC_{\text{norm}}(j) + \Delta PC_{\text{norm}}(j) \quad $$

TDS is defined as the cycle $$j$$ where $$f (PC,TC) $$ reaches its maximum:

$$ TDS = \arg\max_j [f (PC,TC)] \quad $$

# 4. Experiment-ODS
The entire experiment consists of three steps:
- Design and train the Long Short-Term Memory (LSTM) model in source domain.
- Use limited data in target domain to fine-tune the pre-trained model.
- Vary the amount of data available in the target domain and observe its impact on prediction accuracy.

Specifically, we take the THU dataset as an example, where batteries at 55°C serve as the source domain, while batteries at 25°C serve as the target domain. The processing steps for other datasets are similar, aligning feature categories across different datasets. The extracted features are presented in the second section. Additionally, the code for other datasets is stored in the corresponding folder.

## 4.1 Source domain data processing
We apply LSTM networks to predict battery capacity based on historical feature data. The dataset consists of voltage response and other degradation-related features extracted from battery cycling tests. The goal is to preprocess the data, convert it into a supervised learning format, and train an LSTM model for accurate capacity prediction.

The dataset is read from an Excel file, where battery cycling data for different batteries is stored. Two specific battery types (B26T55 and B1T25) are used, and a set of good features with better transferable capability and predictive capability are selected:
```python
file_path = r'Readme-Data.xlsx'  
df = pd.read_excel(file_path, sheet_name='Exp-1')
# Select battery data
B26T55 = df[df['Battery_Name'] == 'B26T55']
# Selected features for model input
feature_26_T55 = pd.concat([
    B26T55['Vg1'], B26T55['Vg9'], B26T55['RVg'],
    B26T55['Q1'], B26T55['Q2'], B26T55['Q3'], B26T55['Q4'], B26T55['Q5'], B26T55['Q6'], B26T55['Q7'], B26T55['Q9'],
    B26T55['RL1'], B26T55['RL8'], B26T55['RL9'], B26T55['RO8']
], axis=1)
```
To train an LSTM model, we must convert sequential data into a supervised learning format, where past observations (n_in time steps) are used to predict future values (n_out time steps):
```python
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
```
Since LSTM networks perform best with normalized data, MinMax scaling is applied to transform feature values between 0 and 1:
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaledFeature_26_T55 = scaler.fit_transform(feature_26_T55)
scaledFeature_26_T55 = pd.DataFrame(data=scaledFeature_26_T55)
```
Converting feature data to time series:
```python
n_steps_in = 3  # Historical time steps
n_steps_out = 1 # Prediction time steps
processedFeature_26_T55 = time_series_to_supervised(scaledFeature_26_T55, n_steps_in, n_steps_out)
```
MinMax scaling is applied to transform capacity values between 0 and 1:
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_26_T55 = scaler.fit_transform(pd.DataFrame(B26T55['Capacity']))
```
Converting capacity data to time series:
```python
processedCapacity_26_T55 = time_series_to_supervised(scaledCapacity_26_T55,n_steps_in,n_steps_out)
```
The processed data is then split into training and testing sets:
```python
# Slice features and labels
data_x26 = processedFeature_26_T55.loc[:,'0(t-3)':'14(t-1)']
data_y26=processedCapacity_26_T55.loc[:,'0']
data_y26=data_y26.values.reshape(-1,1)
# Split training set
train_X26=data_x26.values[:899]
train_y26=data_y26[:899]
train_X26 = train_X26.reshape((train_X26.shape[0], n_steps_in, 15))
```
## 4.2 Train LSTM model
To predict battery capacity, a LSTM network is implemented. LSTM is well-suited for time-series forecasting as it can capture long-term dependencies in sequential data. The LSTM model consists of the following layers: 
- LSTM Layer 1: 96 units, returns sequences (return_sequences=True) to pass outputs to the next LSTM layer.
- LSTM Layer 2: 64 units, does not return sequences (return_sequences=False) as it is the final recurrent layer.
- Dense Layer: 32 neurons with Glorot Uniform initialization.
- Output Layer: 1 neuron for predicting battery capacity.
The model is trained using Mean Squared Error (MSE) as the loss function and the Adam optimizer. The dataset is processed for 75 epochs, meaning it undergoes 75 complete iterations.
Each batch consists of 64 samples, which are processed before updating weights. Shuffling is disabled, ensuring the time-series order is preserved.
```python
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
```
## 4.3 Fine-tune the pre-trained LSTM model
As with the source domain data, we first normalize the target domain data and then convert it to time series data:
```python
# Select features
feature_1_T25=pd.concat([B1T25['Vg1'],B1T25['Vg9'],B1T25['RVg'],
                         B1T25['Q1'],B1T25['Q2'],B1T25['Q3'],B1T25['Q4'],B1T25['Q5'],B1T25['Q6'],B1T25['Q7'],B1T25['Q9'],
                         B1T25['RL1'],B1T25['RL8'],B1T25['RL9'],B1T25['RO8']],axis=1)

# Replace the index in front of B1T25 with starting from 0
B1T25.reset_index(drop=True, inplace=True)

# Normalization
scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_1_T25 = scaler.fit_transform(feature_1_T25)
scaledFeature_1_T25 = pd.DataFrame(data=scaledFeature_1_T25)
# Translate T25 features into time series
n_steps_in =3 
n_steps_out=1
processedFeature_1_T25 = time_series_to_supervised(scaledFeature_1_T25,n_steps_in,n_steps_out)
# Normalize the capacity of T25 and convert it into a time series
# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_1_T25 = scaler.fit_transform(pd.DataFrame(B1T25['Capacity']))
processedCapacity_1_T25 = time_series_to_supervised(scaledCapacity_1_T25,n_steps_in,n_steps_out)
# Slice
data_x1 = processedFeature_1_T25.loc[:,'0(t-3)':'14(t-1)']
data_y1=processedCapacity_1_T25.loc[:,'0']
data_y1=data_y1.values.reshape(-1,1)
```
Divide the training set and testing set according to train_samples. Assuming train_samples is available data. The data before train_samples is used for retraining, and the data after train_samples is used for testing.
```python
train_samples=80 # Change values
train_X1=data_x1.values[:train_samples]
test_X1=data_x1.values[train_samples:]
train_y1=data_y1[:train_samples]
test_y1=data_y1[train_samples:]
train_X1 = train_X1.reshape((train_X1.shape[0], n_steps_in, 15))
test_X1 = test_X1.reshape((test_X1.shape[0], n_steps_in, 15))
```
A fine-tuning process is applied where the model trained on B26T55 (55°C) is adapted for B1T25 (25°C). The first two LSTM layers are frozen to retain previously learned features, and only the dense layers are retrained on the new dataset. The model's Mean Absolute Percentage Error (MAPE) is computed based on true capacity and predicted capacity. The predictions are plotted against the true capacity to assess performance.
```python
# Freeze the first two LSTM layers
for layer in model.layers[:2]:
    layer.trainable = False

# Create a new output layer
input_layer = Input(shape=(train_X1.shape[1], train_X1.shape[2]))
lstm_output_1 = model.layers[0](input_layer)
lstm_output_2 = model.layers[1](lstm_output_1)
new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)

# Create and compile the new model
transfer_model = Model(inputs=input_layer, outputs=new_output_layer)
transfer_model.compile(loss='mse', optimizer='adam')

# Fine-tune the model on the target dataset
start_time = time.time()
transfer_model.fit(train_X1, train_y1, epochs=20, batch_size=64, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)
end_time = time.time()
print(f"Fine-tuning Time: {end_time - start_time} seconds")
# Prediction and model evaluation
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

yhat1t = transfer_model.predict(test_X1)
test_y1 = test_y1.reshape(-1, 1)

inv_forecast_y1t = scaler.inverse_transform(yhat1t)
inv_test_y1t = scaler.inverse_transform(test_y1)

mape_1t = mape(inv_test_y1t, inv_forecast_y1t)
print('Test MAPE: %.3f' % mape_1t)
```
- Note: The train_samles are changed in the range of [20,200], and the interval is 20. The prediction accuracy in each case is recorded 1-mape. The observation found that the use of early data can achieve higher accuracy, and continue to increase the data does not cause a significant increase in accuracy, thus the optimal amount of data is called ODS.
# 5. Experiment-TDS
We propose PC and TC as key components of data sufficiency to evaluate the ability of features to capture battery aging trends and transfer across different conditions. PC measures the correlation between features and capacity, while TC is defined as the 1-Wasserstein distance between source-domain and target-domain features. Both PC and TC exhibit characteristic trends over the battery’s lifecycle, initially maintaining high values in the early stages before declining over time. Accordingly, we define their combination as TDS, whose peak occurs at the same period as the optimal value of ODS.

Calculate the correlation between features and capacity on the 25°C target domain, defined as PC, and record the evolution trend of PC with the cycles：
```python
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
```
Calculate the 1-Wasserstein distance of features between two domains on 25°C target domain and the 55°C source domain, define it as TC, and record the trend of TC with the cycles：
```python
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
```
PC and TC are the components of TDS. We calculate the normalized PC and TC, along with their variations. Then, we determine the cycle at which their combination reaches its maximum value, defining it as TDS：
```python
# Calculate the average of TC and PC
tc_avg = tc_data.mean(axis=1)  
pc_avg = pc_data.mean(axis=1)  
t_values = tc_data.index + 1  

# Calculate maximum value of f(t) and g(t)
max_f = tc_avg.max()  
max_g = pc_avg.max()  

# Normalize f(t) and g(t)
tc_normalized = tc_avg / max_f  
pc_normalized = pc_avg / max_g  

# Build the objective function
# Objective function：TC[t] + PC[t] + (PC[t] - PC[t+1]) + (TC[t] - TC[t+1])
scores = []
for t in range(len(t_values) - 1):  
    score = tc_normalized[t] + pc_normalized[t] + (pc_normalized[t] - pc_normalized[t + 1]) + (tc_normalized[t] - tc_normalized[t + 1])
    scores.append(score)

scores_series = pd.Series(scores, index=t_values[:-1])
# Find the location of the maximum value
max_score_idx = scores_series[:10].idxmax()  
max_score_value = scores_series[:10].max()   
```
# 6. Access
Correspondence to [Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) or [Lin Su](sul24@mails.tsinghua.edu.cn) and CC Prof. [Xuan Zhang](xuanzhang@sz.tsinghua.edu.cn) and [Guangmin Zhou](guangminzhou@sz.tsinghua.edu.cn) when you use, or have any inquiries.
# 7. Acknowledgements
[Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) and [Lin Su](sul24@mails.tsinghua.edu.cn) at Tsinghua Berkeley Shenzhen Institute prepared the data, designed the model and algorithms, developed and tested the experiments, uploaded the model and experimental code, revised the testing experiment plan, and wrote this instruction document based on supplementary materials.



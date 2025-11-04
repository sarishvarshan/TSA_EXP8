# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 04/11/2025
### Name: Sarish Varshan V
### AIM: 
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```py
# BMW Forecasting using Exponential Smoothing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

# Load BMW dataset
data = pd.read_csv("bmw dataset.csv")

# Automatically detect numeric column to forecast
numeric_cols = data.select_dtypes(include=np.number).columns
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found in the dataset.")
value_col = numeric_cols[0]

print(f"Using column for forecasting: {value_col}")
print("First 5 rows:")
print(data.head())

# Create a time index if no Date column exists
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.set_index('Date', inplace=True)
else:
    data.index = pd.RangeIndex(start=0, stop=len(data), step=1)

# Plot Original BMW Data
plt.figure(figsize=(12, 6))
plt.plot(data[value_col], marker='o', color='blue', label='Original BMW Data')
plt.title('BMW Data Over Time')
plt.xlabel('Time Index')
plt.ylabel(value_col)
plt.legend()
plt.grid()
plt.show()

# Calculate and Plot Moving Averages
rolling_mean_5 = data[value_col].rolling(window=5).mean()
rolling_mean_10 = data[value_col].rolling(window=10).mean()

print("\nRolling mean (window=5) first 10 values:")
print(rolling_mean_5.head(10))

print("\nRolling mean (window=10) first 20 values:")
print(rolling_mean_10.head(20))

plt.figure(figsize=(12, 6))
plt.plot(data[value_col], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)', color='orange')
plt.plot(rolling_mean_10, label='Moving Average (window=10)', color='green')
plt.title('Moving Average of BMW Data')
plt.xlabel('Time Index')
plt.ylabel(value_col)
plt.legend()
plt.grid()
plt.show()

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[[value_col]]).flatten()

# Train-test split (80%-20%)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Fit Exponential Smoothing (additive trend, no seasonality)
model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
fit_model = model.fit()
test_predictions = fit_model.forecast(steps=len(test_data))

# Plot Forecast Results
plt.figure(figsize=(12, 6))
plt.plot(range(len(train_data)), train_data, label='Train Data')
plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_data, label='Test Data', color='orange')
plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_predictions, label='Forecast', color='green')
plt.title('Exponential Smoothing Forecast for BMW Data')
plt.xlabel('Time Index')
plt.ylabel(f'Scaled {value_col}')
plt.legend()
plt.grid()
plt.show()

# Model Evaluation
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

```
### OUTPUT:
Original Data:

<img width="718" height="460" alt="Screenshot 2025-10-25 111306" src="https://github.com/user-attachments/assets/1a0da090-acef-4c6f-ba91-c2c6c65a2d94" />

Moving Average:

<img width="240" height="443" alt="image" src="https://github.com/user-attachments/assets/fb8d5b58-ff87-4fa2-8cea-57db9d3e0b7f" />

Plot Transform Dataset:

<img width="966" height="509" alt="image" src="https://github.com/user-attachments/assets/570357bf-b4fd-44d8-b52b-365b66467094" />

Exponential Smoothing:

<img width="955" height="548" alt="image" src="https://github.com/user-attachments/assets/fbc04372-a1cc-4785-b243-d0400320b8de" />

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.

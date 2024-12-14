# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.metrics import RootMeanSquaredError
import joblib  # For saving the scaler
import random
import os

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

# 1. Load the dataset
data = pd.read_csv("wifi_scan_results.csv")

# Drop rows with SSID 'L2G'
data = data[data['SSID'] != 'L2G'].copy()  # Added line to drop 'L2G' SSID

# Convert 'Timestamp' to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Calculate 'measurement_time' as seconds since the first measurement at each location
data['measurement_time'] = data.groupby(['Location_X', 'Location_Y'])['Timestamp'].transform(
    lambda x: (x - x.min()).dt.total_seconds()
)

# Copy data for model training
data['Location_X'] = -data['Location_X']
df = data.copy()

# Label Encoding for BSSID
le_bssid = LabelEncoder()
df['BSSID_encoded'] = le_bssid.fit_transform(df['BSSID'])

# One-Hot Encoding for Frequency
df = pd.get_dummies(df, columns=['Frequency (GHz)'], prefix='Freq')

# Determine the top N most frequent BSSIDs
top_n = 100  # Adjust based on your dataset
top_bssids = df['BSSID'].value_counts().nlargest(top_n).index.tolist()

# Ensure that the specific BSSIDs are included in top_bssids
specific_bssids = ["70:3A:0E:60:E8:E0", "70:3A:0E:60:E8:F0"]
for bssid in specific_bssids:
    if bssid not in top_bssids:
        top_bssids.append(bssid)

# Filter the dataframe to include only top BSSIDs
df_top = df[df['BSSID'].isin(top_bssids)].copy()

# Pivot RSSI values
wifi_rssi = df_top.pivot_table(
    index='Timestamp',
    columns='BSSID',
    values='RSSI (dBm)',
    aggfunc='mean'
)

# Rename columns to indicate RSSI
wifi_rssi.columns = [f'RSSI_{bssid}' for bssid in wifi_rssi.columns]

# Pivot Frequency values
freq_cols = [col for col in df_top.columns if 'Freq_' in col]
wifi_freq = df_top.pivot_table(
    index='Timestamp',
    columns='BSSID',
    values=freq_cols,
    aggfunc='max'
)

# Flatten MultiIndex columns
wifi_freq.columns = [f"{freq}_{bssid}" for freq, bssid in wifi_freq.columns]

# Combine RSSI and Frequency
wifi_features = pd.concat([wifi_rssi, wifi_freq], axis=1)

# Fill missing RSSI with default value (e.g., -100 dBm)
wifi_features = wifi_features.fillna(-100)

# Fill missing Frequency with 0 (since one-hot)
wifi_features = wifi_features.fillna(0)

# Merge with location data
location = df_top.groupby('Timestamp')[['Location_X', 'Location_Y']].first()
features = wifi_features.merge(location, left_index=True, right_index=True)

# Define input features and target variables
X = features.drop(['Location_X', 'Location_Y'], axis=1)
y = features[['Location_X', 'Location_Y']]

# -------------------------------
# Neural Network Model Training
# -------------------------------

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use in the app
joblib.dump(scaler, 'scaler.joblib')

# Build the neural network model
input_dim = X_train_scaled.shape[1]
model = models.Sequential([
    layers.Dense(128, input_dim=input_dim, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(2, activation='linear')  # Output layer for X and Y
])

# Compile the model with RMSE and MAE as metrics
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', RootMeanSquaredError(name='rmse')]
)

# Train the model
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# Evaluate the model
test_metrics = model.evaluate(X_test_scaled, y_test, verbose=2)
# The evaluate method returns [loss, mae, rmse] based on the metrics specified
test_mse = test_metrics[0]
test_mae = test_metrics[1]
test_rmse = test_metrics[2]

print(f"Test RMSE: {test_rmse}")
print(f"Test MAE: {test_mae}")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate R² score for each target variable and average them
r2_x = r2_score(y_test['Location_X'], y_pred[:, 0])
r2_y = r2_score(y_test['Location_Y'], y_pred[:, 1])
r2_avg = (r2_x + r2_y) / 2

print(f"Test R² for Location_X: {r2_x}")
print(f"Test R² for Location_Y: {r2_y}")
print(f"Average Test R²: {r2_avg}")

# Save the trained model
model.save('model.h5')
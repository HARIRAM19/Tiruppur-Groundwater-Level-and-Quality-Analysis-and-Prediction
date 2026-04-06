# model_building.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, Flatten, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load spatial data
df = pd.read_csv('data/spatial/groundwater_spatial.csv', parse_dates=['timestamp'])
df = df.sort_values(['well_id', 'timestamp'])

# Parameters
lookback = 12  # use 12 months to predict next
forecast_horizon = 1

# Select features for model
time_features = ['water_level', 'ph', 'ec', 'tds', 'turbidity', 'temperature']
spatial_features = [col for col in df.columns if col.startswith('block_')] + ['dist_to_tiruppur_km']
# Also include lat/lon? We'll include them as continuous.
spatial_features.extend(['lat', 'lon'])

all_features = time_features + spatial_features

# Scale temporal features (important for NN)
scaler = StandardScaler()
# Fit on all data (in practice, fit on training only)
df_scaled = df.copy()
df_scaled[time_features] = scaler.fit_transform(df[time_features])
# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')

# Build sequences per well
X, y = [], []
well_ids = []
for well_id, group in df_scaled.groupby('well_id'):
    group = group.sort_values('timestamp')
    values = group[all_features].values
    target = group['water_level'].values  # we predict water level (scaled)
    for i in range(len(group) - lookback - forecast_horizon + 1):
        X.append(values[i:i+lookback])
        y.append(target[i+lookback+forecast_horizon-1])
        well_ids.append(well_id)

X = np.array(X)
y = np.array(y)

# Train/test split (by time: last 20% for test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build CNN-LSTM model
def build_cnn_lstm(input_shape):
    inputs = Input(shape=input_shape)
    # CNN
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    # LSTM
    x = LSTM(50, return_sequences=False)(x)
    x = Dense(25, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

input_shape = (lookback, len(all_features))
model = build_cnn_lstm(input_shape)
model.summary()

# Train
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

# Save model
model.save('models/cnn_lstm_waterlevel.h5')

# Evaluate
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

# Ensemble example: could train multiple models with different architectures and average.
# For simplicity, we'll just note that ensemble can be built similarly.

# Anomaly Detection using Autoencoder (on water level)
# Build a simple LSTM autoencoder to reconstruct normal patterns
def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(32, activation='relu', return_sequences=False)(inputs)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(input_shape[1]))(decoded)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Train autoencoder on normal data (use training set)
autoencoder = build_autoencoder(input_shape)
autoencoder.fit(X_train, X_train, validation_split=0.1, epochs=30, batch_size=32, callbacks=[early_stop], verbose=1)
autoencoder.save('models/autoencoder_anomaly.h5')

# Anomaly score = reconstruction error
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - reconstructions), axis=(1,2))
threshold = np.percentile(mse, 95)  # threshold for anomaly
print(f"Anomaly threshold (95th percentile): {threshold}")

# Save threshold
np.save('models/anomaly_threshold.npy', threshold)

# Fine-tuning: later we can load model and continue training on new data.
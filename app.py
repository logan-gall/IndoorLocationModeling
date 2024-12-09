import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import random
import os

# 1. Set Python's built-in random module seed
random.seed(0)

# 2. Set NumPy's random seed
np.random.seed(0)

# 3. Set TensorFlow's random seed
tf.random.set_seed(0)


# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

# 1. Load the dataset
data = pd.read_csv("wifi_scan_results.csv")

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

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
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
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Test MSE: {test_loss}")
print(f"Test MAE: {test_mae}")

# -------------------------------
# Dash Application
# -------------------------------

# Create a unique identifier for each location
data['Location'] = data.apply(lambda row: f"X:{row['Location_X']}, Y:{row['Location_Y']}", axis=1)

# Prepare the map data
unique_locations = data[['Location_X', 'Location_Y']].drop_duplicates()

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Wi-Fi Signal Strength Dashboard with NN Predictions"

# Mapping of BSSID to frequency for annotations
bssid_freq = {
    "70:3A:0E:60:E8:E0": "2.4 GHz",
    "70:3A:0E:60:E8:F0": "5.8 GHz"
}

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Wi-Fi Signal Strength Analysis with NN Predictions",
            style={'textAlign': 'center', 'color': 'white'}),
    html.Div([
        # Map Graph
        dcc.Graph(
            id='map-graph',
            figure=px.scatter(
                unique_locations,
                x='Location_X',
                y='Location_Y',
                title='Location Map',
                labels={'Location_X': 'Location X', 'Location_Y': 'Location Y'},
                hover_data={'Location_X': True, 'Location_Y': True},
                template='plotly_dark',
                color_discrete_sequence=['#C71585']  # Darker pink for actual locations
            )
        )
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        # Signal Strength Graph
        dcc.Graph(
            id='signal-strength-graph',
            figure=px.line(
                title='Select a location on the map to view RSSI over Time',
                template='plotly_dark'
            )
        ),
        # Frequency Notes
        html.Div([
            html.P([
                html.Strong("Frequency Information:"),
                html.Br(),
                '"70:3A:0E:60:E8:E0" - 2.4 GHz',
                html.Br(),
                '"70:3A:0E:60:E8:F0" - 5.8 GHz'
            ], style={'color': 'white', 'textAlign': 'center', 'fontSize': '14px'})
        ], style={'marginTop': '10px'}),
        # Prediction Output
        html.Div(id='prediction-output',
                 style={'textAlign': 'center', 'marginTop': 20,
                        'color': 'white', 'fontSize': 16})
    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    html.Div(id='selected-location',
             style={'textAlign': 'center', 'marginTop': 20,
                    'color': 'white', 'fontSize': 16})
])

# Callback to update graphs and predictions
@app.callback(
    [Output('signal-strength-graph', 'figure'),
     Output('selected-location', 'children'),
     Output('map-graph', 'figure'),
     Output('prediction-output', 'children')],
    [Input('map-graph', 'clickData')]
)
def update_signal_strength(clickData):
    if clickData is None:
        # Initial empty figure
        fig = px.line(
            title='Select a location on the map to view RSSI over Time',
            template='plotly_dark'
        )
        map_fig = px.scatter(
            unique_locations,
            x='Location_X',
            y='Location_Y',
            title='Location Map',
            labels={'Location_X': 'Location X', 'Location_Y': 'Location Y'},
            hover_data={'Location_X': True, 'Location_Y': True},
            template='plotly_dark',
            color_discrete_sequence=['#C71585']  # Darker pink for actual locations
        )
        return fig, "No location selected.", map_fig, ""

    # Extract clicked location coordinates
    point = clickData['points'][0]
    selected_x = point['x']
    selected_y = point['y']

    # Filter data for the selected location and specific BSSIDs
    filtered_data = data[
        (data['Location_X'] == selected_x) &
        (data['Location_Y'] == selected_y) &
        (data['BSSID'].isin(specific_bssids))
    ]

    if filtered_data.empty:
        fig = px.line(
            title='No data available for the selected location and specified BSSIDs.',
            template='plotly_dark'
        )
        location_text = f"Selected Location: X={selected_x}, Y={selected_y}"
        map_fig = px.scatter(
            unique_locations,
            x='Location_X',
            y='Location_Y',
            title='Location Map',
            labels={'Location_X': 'Location X', 'Location_Y': 'Location Y'},
            hover_data={'Location_X': True, 'Location_Y': True},
            template='plotly_dark',
            color_discrete_sequence=['#C71585']  # Darker pink for actual locations
        )
        # Highlight selected location using graph_objects with green 'X'
        map_fig.add_trace(
            go.Scatter(
                x=[selected_x],
                y=[selected_y],
                mode='markers',
                marker=dict(color='green', size=12, symbol='x'),  # Green 'X'
                name='Selected Location'
            )
        )
        return fig, location_text, map_fig, "No prediction available."

    # Calculate average RSSI per measurement_time for the specific BSSIDs
    rssi_over_time = filtered_data.groupby(['measurement_time', 'BSSID'])['RSSI (dBm)'].mean().reset_index()

    # Create the signal strength plot with two lines
    fig = px.line(
        rssi_over_time,
        x='measurement_time',
        y='RSSI (dBm)',
        color='BSSID',
        title=f'RSSI Over Time at Location X={selected_x}, Y={selected_y}',
        labels={'measurement_time': 'Measurement Time (s)', 'RSSI (dBm)': 'RSSI (dBm)', 'BSSID': 'BSSID'},
        template='plotly_dark',
        color_discrete_map={
            "70:3A:0E:60:E8:E0": "#C71585",  # Darker pink
            "70:3A:0E:60:E8:F0": "green"     # Green
        }
    )

    # Set Y-axis range to keep it constant
    fig.update_yaxes(range=[-61, -37])  # Updated Y-axis range

    # Calculate average RSSI per BSSID
    avg_rssi = rssi_over_time.groupby('BSSID')['RSSI (dBm)'].mean().reset_index()

    # Add average lines as grey dashed lines
    for _, row in avg_rssi.iterrows():
        bssid = row['BSSID']
        avg_value = row['RSSI (dBm)']
        fig.add_trace(
            go.Scatter(
                x=[rssi_over_time['measurement_time'].min(), rssi_over_time['measurement_time'].max()],
                y=[avg_value, avg_value],
                mode='lines',
                line=dict(dash='dash', color='grey'),
                name=f"Avg {bssid}"
            )
        )

    # Add annotations for average values near the legend without boxes
    annotations = []
    for i, row in avg_rssi.iterrows():
        bssid = row['BSSID']
        avg_value = row['RSSI (dBm)']
        frequency = bssid_freq.get(bssid, "Unknown GHz")
        annotation_text = f"Avg {frequency}: {avg_value:.1f} dBm"
        # Position annotations based on user-provided values
        annotations.append(
            dict(
                x=1.5,  # Slightly to the right of the plot
                y=0.05 - 0.05 * i,  # Vertically spaced
                xref='paper',
                yref='paper',
                text=annotation_text,
                showarrow=False,
                align='left',
                font=dict(color='grey', size=12)
                # Removed 'bgcolor', 'bordercolor', 'borderwidth', 'borderpad'
            )
        )

    fig.update_layout(
        annotations=annotations,
        legend=dict(
            x=0.85,
            y=1,
            traceorder='normal',
            font=dict(
                family="sans-serif",
                size=12,
                color="white"
            ),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        )
    )

    # Update layout for better visuals
    fig.update_layout(hovermode='closest')

    location_text = f"Selected Location: X={selected_x}, Y={selected_y}"

    # Update the map figure to highlight the selected point using graph_objects
    map_fig = px.scatter(
        unique_locations,
        x='Location_X',
        y='Location_Y',
        title='Location Map',
        labels={'Location_X': 'Location X', 'Location_Y': 'Location Y'},
        hover_data={'Location_X': True, 'Location_Y': True},
        template='plotly_dark',
        color_discrete_sequence=['#C71585']  # Darker pink for actual locations
    )

    # Highlight selected location using graph_objects with green 'X'
    map_fig.add_trace(
        go.Scatter(
            x=[selected_x],
            y=[selected_y],
            mode='markers',
            marker=dict(color='green', size=12, symbol='x'),  # Green 'X'
            name='Selected Location'
        )
    )

    # -------------------------------
    # Displaying Predicted Locations on the Map
    # -------------------------------
    # Find indices in y_test that match the selected location
    selected_indices = y_test[
        (y_test['Location_X'] == selected_x) &
        (y_test['Location_Y'] == selected_y)
    ].index

    if len(selected_indices) == 0:
        prediction_text = "No predictions available for the selected location."
    else:
        # Get the corresponding features
        X_selected_scaled = X_test_scaled[np.isin(X_test.index, selected_indices)]
        y_selected_actual = y_test.loc[selected_indices]
        # Make predictions
        y_pred = model.predict(X_selected_scaled)
        y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_X', 'Predicted_Y'], index=selected_indices)
        # Compare actual vs predicted
        comparison = pd.concat([y_selected_actual.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
        # Calculate error metrics
        mse_x = mean_squared_error(comparison['Location_X'], comparison['Predicted_X'])
        mae_x = mean_absolute_error(comparison['Location_X'], comparison['Predicted_X'])
        mse_y = mean_squared_error(comparison['Location_Y'], comparison['Predicted_Y'])
        mae_y = mean_absolute_error(comparison['Location_Y'], comparison['Predicted_Y'])
        prediction_text = (f"Prediction Errors at Selected Location:\n"
                           f"Location_X - MSE: {mse_x:.2f}, MAE: {mae_x:.2f}\n"
                           f"Location_Y - MSE: {mse_y:.2f}, MAE: {mae_y:.2f}")

        # Add all predicted locations as pink circles
        # Each prediction is a separate point
        map_fig.add_trace(
            go.Scatter(
                x=y_pred_df['Predicted_X'],
                y=y_pred_df['Predicted_Y'],
                mode='markers',
                marker=dict(color='pink', size=8, symbol='circle'),  # Pink circles
                name='Predicted Locations',
                hoverinfo='text',
                text=[f"Predicted Location: X={px_x:.2f}, Y={px_y:.2f}"
                      for px_x, px_y in zip(y_pred_df['Predicted_X'], y_pred_df['Predicted_Y'])]
            )
        )

    return fig, location_text, map_fig, prediction_text

# Run the Dash app
if __name__ == '__main__':
    app.run_server(
        debug=False,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8080))
    )

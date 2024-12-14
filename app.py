# app.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib  # For loading the scaler
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
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

# Copy data for model usage
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
# Load Pre-trained Model and Scaler
# -------------------------------

# Load the trained model
model = load_model('model.h5')

# Load the scaler
scaler = joblib.load('scaler.joblib')

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

# Expose the Flask server for Gunicorn
server = app.server  # <--- Important: Define 'server' for Gunicorn

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
    rssi_over_time = filtered_data.groupby(['measurement_time', 'BSSID', 'SSID'])['RSSI (dBm)'].mean().reset_index()

    # Create BSSID_label column in the format "{BSSID} ({SSID} {frequency})"
    bssid_to_ssid = filtered_data[['BSSID', 'SSID']].drop_duplicates().set_index('BSSID')['SSID'].to_dict()
    rssi_over_time['BSSID_label'] = rssi_over_time['BSSID'].apply(
        lambda bssid: f"{bssid} ({bssid_to_ssid.get(bssid, 'Unknown')} {bssid_freq.get(bssid, 'Unknown')})"
    )

    # Define color mapping based on BSSID_label
    # Assuming "70:3A:0E:60:E8:E0" is mapped to pink and "70:3A:0E:60:E8:F0" to green
    color_discrete_map = {
        f"{bssid} ({bssid_to_ssid.get(bssid, 'Unknown')} {bssid_freq.get(bssid, 'Unknown')})": "#C71585" if bssid == "70:3A:0E:60:E8:E0" else "green"
        for bssid in specific_bssids
    }

    # Create the signal strength plot with two lines using the new BSSID_label
    fig = px.line(
        rssi_over_time,
        x='measurement_time',
        y='RSSI (dBm)',
        color='BSSID_label',
        title=f'RSSI Over Time at Location X={selected_x}, Y={selected_y}',
        labels={'measurement_time': 'Measurement Time (s)', 'RSSI (dBm)': 'RSSI (dBm)', 'BSSID_label': 'BSSID'},
        template='plotly_dark',
        color_discrete_map=color_discrete_map
    )

    # Set Y-axis range to keep it constant
    fig.update_yaxes(range=[-61, -37])  # Updated Y-axis range

    # Calculate average RSSI per BSSID_label
    avg_rssi = rssi_over_time.groupby('BSSID_label')['RSSI (dBm)'].mean().reset_index()

    # Add average lines as grey dashed lines with modified labels
    for _, row in avg_rssi.iterrows():
        bssid_label = row['BSSID_label']
        avg_value = row['RSSI (dBm)']
        # Extract BSSID from BSSID_label
        bssid = bssid_label.split(' ')[0]
        # Create the average label without SSID and frequency, include average value in parenthesis
        avg_label = f"Avg {bssid} ({avg_value:.1f} dBm)"
        fig.add_trace(
            go.Scatter(
                x=[rssi_over_time['measurement_time'].min(), rssi_over_time['measurement_time'].max()],
                y=[avg_value, avg_value],
                mode='lines',
                line=dict(dash='dash', color='grey'),
                name=avg_label  # Modified label
            )
        )

    # -------------------------------
    # Removed Annotations for Average Lines
    # -------------------------------
    # The following block of code that adds annotations has been removed:
    #
    # # Add annotations for average values near the legend without boxes
    # annotations = []
    # for i, row in avg_rssi.iterrows():
    #     bssid_label = row['BSSID_label']
    #     avg_value = row['RSSI (dBm)']
    #     annotation_text = f"Avg: {avg_value:.1f} dBm"
    #     annotations.append(
    #         dict(
    #             x=1.05,  # Slightly to the right of the plot
    #             y=0.95 - 0.05 * i,  # Vertically spaced
    #             xref='paper',
    #             yref='paper',
    #             text=annotation_text,
    #             showarrow=False,
    #             align='left',
    #             font=dict(color='grey', size=12)
    #         )
    #     )
    #
    # fig.update_layout(
    #     annotations=annotations,
    #     legend=dict(
    #         ...
    #     )
    # )

    # Update layout for better visuals
    fig.update_layout(
        legend=dict(
            x=0.6,
            y=1.4,
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
    # Find indices in y that match the selected location
    selected_indices = y[
        (y['Location_X'] == selected_x) &
        (y['Location_Y'] == selected_y)
    ].index

    if len(selected_indices) == 0:
        prediction_text = "No predictions available for the selected location."
    else:
        # Get the corresponding features
        X_selected = X.loc[selected_indices]
        X_selected_scaled = scaler.transform(X_selected)  # Use the loaded scaler
        y_pred = model.predict(X_selected_scaled)
        y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_X', 'Predicted_Y'], index=selected_indices)
        # Compare actual vs predicted
        comparison = pd.concat([y.loc[selected_indices].reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
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
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

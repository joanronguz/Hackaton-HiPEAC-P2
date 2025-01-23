# -*- coding: utf-8 -*-
# Data Analysis

### Train Data Analysis

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras import layers as tfkl
import numpy as np

# Read the train and validation data

train_df = pd.read_csv('trainData.csv')
val_df = pd.read_csv('validationData.csv')

# Display the first few rows of each dataset
print("Train Data:")
print(train_df.head())

print("\nValidation Data:")
print(val_df.head())

# import pandas as pd

# # Load the uploaded file
# file_path = '/mnt/data/trainData.csv'
# train_df = pd.read_csv(file_path)

# Display basic information about the dataset
train_info = {
    "head": train_df.head(),
    "info": train_df.info(),
    "describe": train_df.describe(include="all"),
    "null_values": train_df.isnull().sum(),
    "columns": train_df.columns.tolist()
}

train_info

# Convert 'data' column to datetime
train_df['data'] = pd.to_datetime(train_df['data'])

# Handle missing values: Fill missing NO2 with the median for each station
train_df['NO2'] = train_df.groupby('nom_estacio')['NO2'].transform(lambda x: x.fillna(x.median()))

# Investigate and handle negative NO2 values (e.g., set them to NaN or replace them with a minimum threshold)
train_df.loc[train_df['NO2'] < 0, 'NO2'] = None
train_df['NO2'] = train_df['NO2'].fillna(train_df['NO2'].median())

# Summary statistics after preprocessing
summary_stats = train_df['NO2'].describe()

# Analyze NO2 levels by station and hour
station_hourly_mean = train_df.groupby(['nom_estacio', 'hour'])['NO2'].mean().reset_index()

# Plot distribution of NO2 levels

# Analyze NO2 trends per station
station_trends = train_df.groupby(['nom_estacio', train_df['data'].dt.month])['NO2'].mean().reset_index()

print(summary_stats)
print(station_hourly_mean.head())

# Ensure 'data' is in datetime format
train_df['data'] = pd.to_datetime(train_df['data'], errors='coerce')

# Combine 'data' and 'hour' into a single datetime column
train_df['datetime'] = train_df['data'] + pd.to_timedelta(train_df['hour'] - 1, unit='h')

# Set up the plot
plt.figure(figsize=(14, 8))

# Plot NO2 levels for each station
stations = train_df['nom_estacio'].unique()
for station in stations:
    station_data = train_df[train_df['nom_estacio'] == station]
    plt.plot(station_data['datetime'], station_data['NO2'], label=station, alpha=0.7)

# Format the x-axis to show both day and hour
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.xticks(rotation=45)

# Labels and legend
plt.title("NO2 Levels Over Time for Each Station")
plt.xlabel("Date and Hour")
plt.ylabel("NO2 Concentration")
plt.legend(title="Station")
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

# Set up subplots for each station
stations = train_df['nom_estacio'].unique()
fig, axes = plt.subplots(len(stations), 1, figsize=(14, 20), sharex=True)

for i, station in enumerate(stations):
    station_data = train_df[train_df['nom_estacio'] == station]
    axes[i].plot(station_data['datetime'], station_data['NO2'], label=f"{station} NO2 Levels", alpha=0.7)
    avg_NO2 = station_data['NO2'].mean()
    axes[i].axhline(avg_NO2, color='red', linestyle='--', label=f"Average NO2 ({avg_NO2:.2f})")
    axes[i].set_title(f"NO2 Levels Over Time for {station}")
    axes[i].set_ylabel("NO2 Concentration")
    axes[i].legend()
    axes[i].grid(True)

# Common x-axis formatting
plt.xlabel("Date and Hour")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Ensure 'data' is in datetime format
train_df['data'] = pd.to_datetime(train_df['data'], errors='coerce')

# Combine 'data' and 'hour' into a single datetime column
train_df['datetime'] = train_df['data'] + pd.to_timedelta(train_df['hour'] - 1, unit='h')

# Set up subplots for each station to visualize missing values
stations = train_df['nom_estacio'].unique()
fig, axes = plt.subplots(len(stations), 1, figsize=(14, 20), sharex=True)

for i, station in enumerate(stations):
    station_data = train_df[train_df['nom_estacio'] == station]
    missing_data = station_data[station_data['NO2'].isnull()]

    axes[i].scatter(
        missing_data['datetime'],
        [0] * len(missing_data),  # Dummy y-values for visualization
        color='red',
        label='Missing Values',
        alpha=0.6
    )

    axes[i].set_title(f"Missing NO2 Values Over Time for {station}")
    axes[i].set_ylabel("Missing Values (Indicator)")
    axes[i].legend()
    axes[i].grid(True)

# Common x-axis formatting
plt.xlabel("Date and Hour")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Check if there are NaN values in the 'NO2' column
print("Missing values per station:")
print(train_df[train_df['NO2'].isnull()]['nom_estacio'].value_counts())

# Ensure 'datetime' column is correctly created
train_df['data'] = pd.to_datetime(train_df['data'], errors='coerce')
train_df['datetime'] = train_df['data'] + pd.to_timedelta(train_df['hour'] - 1, unit='h')

# Filter data for missing NO2 values
missing_data = train_df[train_df['NO2'].isnull()]
if missing_data.empty:
    print("No missing values found for NO2 in the dataset.")
else:
    # Plot missing values per station
    fig, axes = plt.subplots(len(stations), 1, figsize=(14, 20), sharex=True)

    for i, station in enumerate(stations):
        station_data = missing_data[missing_data['nom_estacio'] == station]

        if not station_data.empty:
            axes[i].scatter(
                station_data['datetime'],
                [0] * len(station_data),  # Dummy y-values for visualization
                color='red',
                label='Missing Values',
                alpha=0.6
            )
        else:
            print(f"No missing NO2 values for station: {station}")

        axes[i].set_title(f"Missing NO2 Values Over Time for {station}")
        axes[i].set_ylabel("Missing Values (Indicator)")
        axes[i].legend()
        axes[i].grid(True)

    # Common x-axis formatting
    plt.xlabel("Date and Hour")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Filter missing values and check the stations
missing_data = train_df[train_df['NO2'].isnull()]
stations = train_df['nom_estacio'].unique()

fig, axes = plt.subplots(len(stations), 1, figsize=(14, 20), sharex=True)

for i, station in enumerate(stations):
    station_data = missing_data[missing_data['nom_estacio'] == station]

    if not station_data.empty:
        axes[i].scatter(
            station_data['datetime'],
            [0] * len(station_data),  # Dummy y-values for visualization
            color='red',
            label='Missing Values',
            alpha=0.6
        )
    else:
        axes[i].text(0.5, 0.5, "No Missing Values", transform=axes[i].transAxes, ha="center")

    axes[i].set_title(f"Missing NO2 Values Over Time for {station}")
    axes[i].set_ylabel("Missing Values (Indicator)")
    axes[i].legend()
    axes[i].grid(True)

# Common x-axis formatting
plt.xlabel("Date and Hour")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Reload the original dataset
train_df = pd.read_csv('trainData.csv')

# Ensure 'data' is in datetime format
train_df['data'] = pd.to_datetime(train_df['data'], errors='coerce')

# Combine 'data' and 'hour' into a single datetime column
train_df['datetime'] = train_df['data'] + pd.to_timedelta(train_df['hour'] - 1, unit='h')

# Filter missing values from the original dataset
missing_data = train_df[train_df['NO2'].isnull()]
stations = train_df['nom_estacio'].unique()

# Plot missing values per station
fig, axes = plt.subplots(len(stations), 1, figsize=(14, 20), sharex=True)

for i, station in enumerate(stations):
    station_data = missing_data[missing_data['nom_estacio'] == station]

    if not station_data.empty:
        axes[i].scatter(
            station_data['datetime'],
            [0] * len(station_data),  # Dummy y-values for visualization
            color='red',
            label='Missing Values',
            alpha=0.6
        )
    else:
        axes[i].text(0.5, 0.5, "No Missing Values", transform=axes[i].transAxes, ha="center")

    axes[i].set_title(f"Missing NO2 Values Over Time for {station}")
    axes[i].set_ylabel("Missing Values (Indicator)")
    axes[i].legend()
    axes[i].grid(True)

# Common x-axis formatting
plt.xlabel("Date and Hour")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Count the number of missing NO2 values for each station
missing_per_station = train_df[train_df['NO2'].isnull()]['nom_estacio'].value_counts()

# Display the counts
print("Missing NO2 values per station:")
print(missing_per_station)


train_df = pd.read_csv('trainData.csv')

# Count missing and non-missing NO2 values per station
missing_counts = train_df.groupby('nom_estacio')['NO2'].apply(lambda x: x.isnull().sum())
existing_counts = train_df.groupby('nom_estacio')['NO2'].apply(lambda x: x.notnull().sum())

# Create a DataFrame for fractions
fraction_df = pd.DataFrame({
    'Missing': missing_counts,
    'Existing': existing_counts
})
fraction_df['Total'] = fraction_df['Missing'] + fraction_df['Existing']
fraction_df['Missing Fraction'] = fraction_df['Missing'] / fraction_df['Total']
fraction_df['Existing Fraction'] = fraction_df['Existing'] / fraction_df['Total']

# Plot the fractions as a bar chart
fraction_df[['Missing Fraction', 'Existing Fraction']].plot(
    kind='bar',
    stacked=True,
    figsize=(10, 6),
    color=['red', 'green'],
    alpha=0.7
)

plt.title('Fraction of Missing and Existing NO2 Values per Station')
plt.ylabel('Fraction')
plt.xlabel('Station')
plt.xticks(rotation=45)
plt.legend(title='Value Type')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Create a bar plot for real numbers of missing and existing values
fig, ax = plt.subplots(figsize=(10, 6))

# Bar width for side-by-side bars
bar_width = 0.4
stations = fraction_df.index
x = range(len(stations))

# Plot the data
ax.bar(x, fraction_df['Existing'], width=bar_width, label='Existing', color='green', alpha=0.7)
ax.bar([pos + bar_width for pos in x], fraction_df['Missing'], width=bar_width, label='Missing', color='red', alpha=0.7)

# Add labels and legend
ax.set_xticks([pos + bar_width / 2 for pos in x])
ax.set_xticklabels(stations, rotation=45)
ax.set_ylabel('Count')
ax.set_title('Count of Missing and Existing NO2 Values per Station')
ax.legend(title='Value Type')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

"""### Validation Data Analysis"""

# Display basic information about the dataset
val_info = {
    "head": val_df.head(),
    "info": val_df.info(),
    "describe": val_df.describe(include="all"),
    "null_values": val_df.isnull().sum(),
    "columns": val_df.columns.tolist()
}

val_info

# # Load the validation dataset
# val_file_path = '/mnt/data/validationData.csv'
# val_df = pd.read_csv(val_file_path)

# Count missing and existing NO2 values
val_missing_count = val_df['NO2'].isnull().sum()
val_existing_count = val_df['NO2'].notnull().sum()

# Create a bar plot for missing and existing values
fig, ax = plt.subplots(figsize=(6, 6))

# Bar labels
labels = ['Existing', 'Missing']
values = [val_existing_count, val_missing_count]
colors = ['green', 'red']

# Plot the data
ax.bar(labels, values, color=colors, alpha=0.7)

# Add labels and title
ax.set_ylabel('Count')
ax.set_title('Count of Missing and Existing NO2 Values (Validation)')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Ensure 'data' is in datetime format
val_df['data'] = pd.to_datetime(val_df['data'], errors='coerce')

# Combine 'data' and 'hour' into a single datetime column
val_df['datetime'] = val_df['data'] + pd.to_timedelta(val_df['hour'] - 1, unit='h')

# Set up a single plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot NO2 levels over time
ax.plot(val_df['datetime'], val_df['NO2'], label='NO2 Levels', alpha=0.7)

# Calculate and plot the average NO2 as a horizontal line
avg_NO2 = val_df['NO2'].mean()
ax.axhline(avg_NO2, color='red', linestyle='--', label=f"Average NO2 ({avg_NO2:.2f})")

# Add title, labels, and legend
ax.set_title("NO2 Levels Over Time (Validation Dataset)")
ax.set_xlabel("Date and Hour")
ax.set_ylabel("NO2 Concentration")
ax.legend()
ax.grid(True)

# Format x-axis for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""## TrainData : dealing with `NaN`"""

# Load the dataset to examine it
file_path = 'trainData.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()

# Select only numeric columns for negative value counting
numeric_data = data.select_dtypes(include=['number'])

# Count negative values for each station (column)
negative_count = (numeric_data < 0).sum()

# Combine the results in a dataframe for display
summary = pd.DataFrame({
    'NaN Count': nan_count,
    'Negative Count': negative_count
})

#tools.display_dataframe_to_user(name="NaN and Negative Values Summary", dataframe=summary)

# Replace negative values with NaN in the numeric columns
numeric_data[numeric_data < 0] = float('nan')

# Count NaN values again after replacement
nan_count_updated = numeric_data.isna().sum()

# Combine the updated results in a dataframe for display
summary_updated = pd.DataFrame({
    'NaN Count': nan_count_updated,
    'Negative Count': negative_count  # Using the previous negative count
})


# Plot the processed dataset (replace negative values with NaN) as a temporary series for each station
# We will use the numeric data after replacing negative values
processed_data_series = numeric_data.mean()  # Plotting the mean of each column as a simple series

# Convert the 'data' column to datetime and create a 'datetime' column combining 'data' and 'hour'
data['datetime'] = pd.to_datetime(data['data']) + pd.to_timedelta(data['hour'] - 1, unit='h')

# Plotting NO2 concentration as a time series for each station
stations = data['nom_estacio'].unique()

plt.figure(figsize=(12, 6))

for station in stations:
    station_data = data[data['nom_estacio'] == station]
    plt.plot(station_data['datetime'], station_data['NO2'], label=station)

# Re-load the dataset and preprocess the NO2 column by replacing negative values with NaN
data = pd.read_csv(file_path)

# Create a new processed version of the data where negative NO2 values are replaced with NaN
data_processed = data.copy()
data_processed['NO2'] = data_processed['NO2'].apply(lambda x: x if x >= 0 else float('nan'))

# Create datetime column by combining 'data' (date) and 'hour'
data_processed['datetime'] = pd.to_datetime(data_processed['data']) + pd.to_timedelta(data_processed['hour'] - 1, unit='h')

# Now plot the processed data (1 plot for each of the 5 stations)
stations = data_processed['nom_estacio'].unique()

plt.figure(figsize=(12, 8))

# Generate a plot for each station
for i, station in enumerate(stations, start=1):
    plt.subplot(3, 2, i)  # Create a subplot for each station
    station_data = data_processed[data_processed['nom_estacio'] == station]
    plt.plot(station_data['datetime'], station_data['NO2'], label=station)
    plt.title(f'NO2 Concentration - {station}')
    plt.xlabel('Time')
    plt.ylabel('NO2 Concentration')
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()

# Step 1: Round the datetime to the nearest hour for consistency (if necessary)
data_processed['datetime_hour'] = data_processed['datetime'].dt.floor('H')

# Step 2: Compute the mean NO2 concentration for each unique hour across all stations
hourly_mean = (
    data_processed.groupby('datetime_hour')['NO2']
    .mean()
    .reset_index()
    .rename(columns={'NO2': 'NO2_mean_hourly'})
)

# Step 3: Extract the date from the datetime_hour column
hourly_mean['date'] = hourly_mean['datetime_hour'].dt.date

# Step 4: Optionally reduce to one row per day (daily mean of hourly means)
daily_mean = (
    hourly_mean.groupby('date')['NO2_mean_hourly']
    .mean()
    .reset_index()
    .rename(columns={'NO2_mean_hourly': 'NO2_daily_mean'})
)

# Step 1: Round the datetime to the nearest hour for consistency
data_processed['datetime_hour'] = data_processed['datetime'].dt.floor('H')

# Step 2: Compute the mean NO2 concentration for each unique hour across all stations
hourly_mean = (
    data_processed.groupby('datetime_hour')['NO2']
    .mean()
    .reset_index()
    .rename(columns={'NO2': 'NO2_mean_hourly'})
)

# Step 3: Replace the original dataset with the reduced dataset
data_processed = hourly_mean.copy()

# Step 4: Verify the reduced dataset
print(data_processed.head())

# Count NaN values in the NO2 column of the processed data
nan_count = data_processed['NO2_mean_hourly'].isna().sum()

print(f"Number of NaN values in NO2 column: {nan_count}")

# Count valid (non-NaN) and NaN values for the NO2 column
valid_count = data_processed['NO2_mean_hourly'].notna().sum()
nan_count = data_processed['NO2_mean_hourly'].isna().sum()

# Perform linear interpolation to fill NaN values in the NO2 column
data_processed['NO2_interpolated'] = data_processed['NO2_mean_hourly'].interpolate(method='linear')

# Perform linear interpolation to fill NaN values in the NO2 column
data_processed['NO2_interpolated'] = data_processed['NO2_mean_hourly'].interpolate(method='linear')

# Count NaN values after interpolation (should be 0 if interpolation worked correctly)
nan_count_interpolated = data_processed['NO2_interpolated'].isna().sum()

nan_count_interpolated

# Count valid (non-NaN) and NaN values for the interpolated NO2 column
valid_count_interpolated = data_processed['NO2_interpolated'].notna().sum()
nan_count_interpolated = data_processed['NO2_interpolated'].isna().sum()

# Save the preprocessed dataset to a CSV file in the current directory
preprocessed_file_path = 'preprocessed_data.csv'
data_processed.to_csv(preprocessed_file_path, index=False)

preprocessed_file_path

"""## ValData : dealing with `NaN`"""

# Load the validation dataset
validation_data_path = 'validationData.csv'
validation_data = pd.read_csv(validation_data_path)

# Perform the same preprocessing on the validation data (replace negative values with NaN, then interpolate)
validation_data['NO2'] = validation_data['NO2'].apply(lambda x: x if x >= 0 else float('nan'))
validation_data['NO2_interpolated'] = validation_data['NO2'].interpolate(method='linear')

# Count NaN values after interpolation for validation data
nan_count_validation_interpolated = validation_data['NO2_interpolated'].isna().sum()

nan_count_validation_interpolated

# Save the preprocessed validation dataset to a CSV file
preprocessed_validation_file_path = 'preprocessed_validation_data.csv'
validation_data.to_csv(preprocessed_validation_file_path, index=False)

preprocessed_validation_file_path

"""## Deep Learning Model"""

# Read the train and validation data
train_data_processed = pd.read_csv('preprocessed_data.csv')
validation_data_processed = pd.read_csv('preprocessed_validation_data.csv')

# Display the first few rows of the train and validation datasets
train_head = train_data_processed.head()
validation_head = validation_data_processed.head()

train_head, validation_head
# Remove the 'NO2' column and retain 'NO2_interpolated'
train_data_processed = train_data_processed.drop(columns=['NO2_mean_hourly'])

# Display the updated validation dataset's head
print("Updated Train Dataset:")
print(train_data_processed.head())
print("\nShape of the Train Dataset:", train_data_processed.shape)

# Remove the 'NO2' column and retain 'NO2_interpolated'
validation_data_processed = validation_data_processed.drop(columns=['NO2'])

# Display the updated validation dataset's head
print("Updated Validation Dataset:")
print(validation_data_processed.head())
print("\nShape of the Val Dataset:", validation_data_processed.shape)

def create_sliding_windows(
    series: np.ndarray,
    input_length: int = 168,
    forecast_length: int = 24
):
    """
    Given a 1D or 2D NumPy array `series` of shape (time, features),
    return X of shape (num_samples, input_length, features) and
    y of shape (num_samples, forecast_length, features).

    If the series is 1D, it will be reshaped to (time, 1).
    """
    if series.ndim == 1:
        # Reshape to (time, 1)
        series = series.reshape(-1, 1)

    X_list, y_list = [], []

    max_start = len(series) - input_length - forecast_length + 1
    for start_idx in range(max_start):
        end_idx = start_idx + input_length
        forecast_end_idx = end_idx + forecast_length

        X_window = series[start_idx:end_idx]
        y_window = series[end_idx:forecast_end_idx]

        X_list.append(X_window)
        y_list.append(y_window)

    return np.array(X_list), np.array(y_list)

train_series = train_data_processed['NO2_interpolated'].values

# Choose window sizes
input_length = 168
forecast_length = 24

X_train, y_train = create_sliding_windows(
    series=train_series,
    input_length=input_length,
    forecast_length=forecast_length
)

print("X_train shape:", X_train.shape)  # (N, 168, 1) if only NO2
print("y_train shape:", y_train.shape)  # (N, 24, 1)

val_series = validation_data_processed['NO2_interpolated'].values

X_val, y_val = create_sliding_windows(
    series=val_series,
    input_length=input_length,
    forecast_length=forecast_length
)

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

import tensorflow as tf
from tensorflow.keras import layers as tfkl

def build_CONV_LSTM_model(input_shape, output_shape):
    # Ensure the input time steps are at least as many as the output time steps
    assert input_shape[0] >= output_shape[0], \
        "For this exercise we want input time steps >= output time steps"

    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    # 1st LSTM layer
    x = tfkl.LSTM(128, return_sequences=True, name='lstm1')(input_layer)
    x = tfkl.Dropout(0.3)(x)

    # 2nd LSTM layer
    x = tfkl.LSTM(128, return_sequences=True, name='lstm2')(x)
    x = tfkl.Dropout(0.3)(x)

    # 1D Convolution + ReLU
    x = tfkl.Conv1D(128, 3, padding='same', name='conv1')(x)
    x = tfkl.Activation('relu', name='relu_after_conv1')(x)
    x = tfkl.Dropout(0.3)(x)

    # 1D Convolution + ReLU
    x = tfkl.Conv1D(128, 3, padding='same', name='conv2')(x)
    x = tfkl.Activation('relu', name='relu_after_conv2')(x)
    x = tfkl.Dropout(0.3)(x)

    # Final Convolution => matches desired output's features
    output_layer = tfkl.Conv1D(output_shape[1], 3, padding='same', name='output_layer')(x)

    # Crop the time dimension to match output_shape[0]
    crop_size = output_layer.shape[1] - output_shape[0]
    output_layer = tfkl.Cropping1D((0, crop_size), name='cropping')(output_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='CONV_LSTM_model')
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.AdamW()
    )
    return model

# Build it with the shapes you have
input_shape = (168, 1)  # 168 hours, 1 feature if you're only using NO2
output_shape = (24, 1)  # 24 hours forecast, 1 feature

model = build_CONV_LSTM_model(input_shape, output_shape)
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,               # Increase as needed
    batch_size=32,           # Adjust to your system's memory
    validation_data=(X_val, y_val),
    verbose=1
)
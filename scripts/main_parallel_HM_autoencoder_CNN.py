# --- Libraries ---
from utils.helpers import interpolate_time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import torch
import os
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
import tensorflow as tf # Import tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# --- Configuration & Data Reading ---
start_date = pd.to_datetime("2023-10-01")
SPLIT = 0.60
path_to_embeddings = "data/embeddings/autoencoder"
path_to_all_images = 'data/microscope_images' 

# Read Lab Measurements
lab_measurements = pd.read_csv("data/lab_measurements.txt", sep='\s+', index_col=0)
lab_measurements = lab_measurements.iloc[1:].astype(float)
lab_measurements.index = pd.to_numeric(lab_measurements.index, errors='coerce')
lab_measurements.index = start_date + pd.to_timedelta(lab_measurements.index, unit='D')
lab_measurements = lab_measurements.interpolate(method='time')

# Read Mechanistic Model Output
mech_model_output = pd.read_csv("data/Project1.Dynamic.Simul.out.txt", sep='\s+', index_col=0)
mech_model_output = mech_model_output.iloc[1:].astype(float)
mech_model_output.index = pd.to_numeric(mech_model_output.index, errors='coerce')
mech_model_output.index = start_date + pd.to_timedelta(mech_model_output.index, unit='D')

# Get Sorted Folders (use actual image path as reference)
all_image_folders = sorted(listdir(path_to_all_images))
num_folders_total = len(all_image_folders)

# Interpolate Measurements/Model Output to Folder Dates
lab_measurements_reindex = interpolate_time(lab_measurements, all_image_folders)
mech_model_output_reindex = interpolate_time(mech_model_output, all_image_folders)

# Calculate Errors (length = number of folders/time points)
TSS_eff_error = lab_measurements_reindex['TSSeffluent'].values - mech_model_output_reindex['.SST_1.X_Out'].values
TSS_ras_error = lab_measurements_reindex['TSSras'].values - mech_model_output_reindex['.SST_1.X_Under'].values

# --- Load ALL Features and Create Corresponding Labels and Folder Mapping ---
all_features_list = []
all_labels_eff_list = []
all_labels_ras_list = []
feature_folder_map = [] # Keep track of which folder each feature belongs to
processed_folder_indices = [] # Keep track of folders we actually found data for

folder_idx_counter = 0 
for folder in all_image_folders:
    # Path to embeddings for this folder
    path_to_embeddings_folder = f"{path_to_embeddings}/{folder}/basin5/10x"
    
    if not os.path.exists(path_to_embeddings_folder) or not listdir(path_to_embeddings_folder):
        print(f"Warning: Embeddings path not found for folder {folder}, skipping.")
        folder_idx_counter += 1 # Still increment error index if skipping folder
        continue

    images_list_embeddings = listdir(path_to_embeddings_folder)
    images_in_folder_count = 0

    for image_file in images_list_embeddings:
        try:
            img_path = f"{path_to_embeddings_folder}/{image_file}"
            img = torch.load(img_path).cpu().numpy()
            all_features_list.append(img)
            # Assign labels corresponding to this FOLDER 
            all_labels_eff_list.append(TSS_eff_error[folder_idx_counter])
            all_labels_ras_list.append(TSS_ras_error[folder_idx_counter])
            feature_folder_map.append(len(processed_folder_indices)) # Store the folder index for this feature
            images_in_folder_count += 1
        except Exception as e:
            print(f"Error loading or processing {img_path}: {e}")
    
    if images_in_folder_count > 0:
        processed_folder_indices.append(folder_idx_counter) # Add original index if processed

    folder_idx_counter += 1 # Move to the next folder's error value

all_features = np.array(all_features_list)
TSSeffluent_labels = np.array(all_labels_eff_list)
TSSras_labels = np.array(all_labels_ras_list) 
feature_folder_map = np.array(feature_folder_map)

print(f"Total folders found: {len(all_image_folders)}")
print(f"Folders successfully processed: {len(processed_folder_indices)}")
print(f"Total image features loaded: {len(all_features)}")
print(f"Labels array length (Effluent): {len(TSSeffluent_labels)}")
print(f"Feature matrix shape: {all_features.shape}")
assert len(all_features) == len(TSSeffluent_labels) # Check features match labels
assert len(lab_measurements_reindex) == len(processed_folder_indices) # Check reindexed length

# --- Split Based on Time (Folders) ---
split_folder_index = round(SPLIT * len(processed_folder_indices))

# Get indices of features belonging to train folders vs test folders
train_indices = np.where(feature_folder_map < split_folder_index)[0]
test_indices = np.where(feature_folder_map >= split_folder_index)[0]

# Create train/test sets
X_train = all_features[train_indices]
y_train_eff = TSSeffluent_labels[train_indices]
X_test = all_features[test_indices]
y_test_eff = TSSeffluent_labels[test_indices]

y_train_ras = TSSras_labels[train_indices]
y_test_ras = TSSras_labels[test_indices]

print(f"Splitting at processed folder index: {split_folder_index} (out of {len(processed_folder_indices)})")
print(f"Train set size: {len(train_indices)} images")
print(f"Test set size: {len(test_indices)} images")

# --- Transpose for Keras ---
# Transpose AFTER splitting
X_train = X_train.transpose(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
X_test = X_test.transpose(0, 2, 3, 1)
all_features_transposed = all_features.transpose(0, 2, 3, 1) 

# Check input shape matches X_train
input_shape_keras = X_train.shape[1:] # (H, W, C)
print(f"Input shape for Keras model: {input_shape_keras}")

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_keras))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # Dropout rate (e.g., 50%), tune as needed
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()]) # Add MAE metric
model.summary()

# --- Train the Model ---
history = model.fit(X_train, y_train_eff, epochs=20, batch_size=32,
                    validation_split=0.2, # validates on last 20% of X_train
                    verbose=1)

# --- Evaluate on Explicit Test Set ---
print("\nEvaluating on Test Set:")
test_loss, test_mae = model.evaluate(X_test, y_test_eff, verbose=0)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test Mean Absolute Error: {test_mae:.4f}")

# --- Predict on ALL Data for Plotting Hybrid Model ---
y_pred_all = model.predict(all_features_transposed) 
y_pred_all = y_pred_all.squeeze() # (N, 1)-> (N,)

# --- Average Predictions per Time Point (Folder) ---
average_eff_error_preds = []
std_dev = []
current_pred_index = 0
for folder_idx in range(len(processed_folder_indices)): 
    num_images_in_folder = np.sum(feature_folder_map == folder_idx)

    if num_images_in_folder > 0:
        # Get the predictions corresponding to this folder
        preds_for_folder = y_pred_all[current_pred_index : current_pred_index + num_images_in_folder]
        average_eff_error_preds.append(np.median(preds_for_folder)) # Use median as before
        std_dev.append(np.std(preds_for_folder))
        current_pred_index += num_images_in_folder
    else:
        print(f"Warning: No predictions found for folder index {folder_idx}. Appending NaN.")
        average_eff_error_preds.append(np.nan) 
        std_dev.append(np.nan)

average_eff_error_preds = np.array(average_eff_error_preds)
std_dev = np.array(std_dev)

# --- Calculate Hybrid Model and Uncertainty ---
# Ensure index alignment - crucial if folders were skipped
eff_TSS_hybridmodel = pd.Series(
    mech_model_output_reindex['.SST_1.X_Out'].values + average_eff_error_preds,
    index=mech_model_output_reindex.index
)

eff_TSS_hybridmodel_upper = pd.Series(
    eff_TSS_hybridmodel.values + std_dev,
    index=mech_model_output_reindex.index
)
eff_TSS_hybridmodel_lower = pd.Series(
    eff_TSS_hybridmodel.values - std_dev,
    index=mech_model_output_reindex.index
)

# --- Plotting ---
# Use the FOLDER-based split index for plotting
plot_split_index = split_folder_index 

plt.figure(figsize=(10, 4), dpi=150)
plt.plot(lab_measurements_reindex['TSSeffluent'], '.-', label='Measurements', color='blue')
plt.plot(mech_model_output_reindex['.SST_1.X_Out'], '-', linewidth=1.5, label='Mechanistic model output', color='green')

# Plot hybrid model predictions - use iloc for positional slicing based on plot_split_index
plt.plot(eff_TSS_hybridmodel.iloc[:plot_split_index], '.-', label='HM predictions (train)', color='orange')
plt.plot(eff_TSS_hybridmodel.iloc[plot_split_index:], '.-', label='HM predictions (test)', color='red')

# Plot Standard Deviation Band (Train) - use iloc
plt.fill_between(eff_TSS_hybridmodel.index[:plot_split_index],
                 eff_TSS_hybridmodel_lower.iloc[:plot_split_index],
                 eff_TSS_hybridmodel_upper.iloc[:plot_split_index],
                 color='orange', alpha=0.2, zorder=1)

# Plot Standard Deviation Band (Test) - use iloc
plt.fill_between(eff_TSS_hybridmodel.index[plot_split_index:],
                 eff_TSS_hybridmodel_lower.iloc[plot_split_index:],
                 eff_TSS_hybridmodel_upper.iloc[plot_split_index:],
                 color='red', alpha=0.2, zorder=1)

plt.xlabel("Time")
plt.ylabel("TSS effluent (mg/L)")
plt.legend()
plt.show()

############################################################################ TSS ras
# Define the CNN model
model2 = Sequential()
model2.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape_keras))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(8, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(64, activation='linear'))
model2.add(Dropout(0.5)) # Dropout rate
model2.add(Dense(1))

# Compile the model
model2.compile(optimizer=Adam(learning_rate=0.0001), loss='mae', metrics=[tf.keras.metrics.MeanAbsoluteError()]) # Add MAE metric
model2.summary()

# --- Train the Model ---
history = model2.fit(X_train, y_train_ras, epochs=25, batch_size=32,
                    validation_split=0.2, # validates on last 20% of X_train
                    verbose=1)

# --- Evaluate Model 2 on Explicit Test Set ---
print("\nEvaluating RAS Model on Test Set:")
test_loss_ras, test_mae_ras = model2.evaluate(X_test, y_test_ras, verbose=0)
print(f"Test Loss (MAE): {test_loss_ras:.4f}")
print(f"Test Mean Absolute Error: {test_mae_ras:.4f}")

# --- Predict on ALL Data for Plotting Hybrid Model ---
y_pred_ras_all = model2.predict(all_features_transposed)
y_pred_ras_all = y_pred_ras_all.squeeze()

# --- Average Predictions per Time Point (Folder) ---
average_ras_error_preds = []
std_dev_ras = []
current_pred_index = 0
for folder_idx in range(len(processed_folder_indices)): 
    # Find how many features belong to this folder
    num_images_in_folder = np.sum(feature_folder_map == folder_idx)

    if num_images_in_folder > 0:
        # Get the predictions corresponding to this folder
        preds_for_folder = y_pred_ras_all[current_pred_index : current_pred_index + num_images_in_folder]
        average_ras_error_preds.append(np.median(preds_for_folder)) 
        std_dev_ras.append(np.std(preds_for_folder))
        current_pred_index += num_images_in_folder
    else:
        print(f"Warning: No RAS predictions found for folder index {folder_idx}. Appending NaN.")
        average_ras_error_preds.append(np.nan)
        std_dev_ras.append(np.nan)

average_ras_error_preds = np.array(average_ras_error_preds)
std_dev_ras = np.array(std_dev_ras)

# --- Create Hybrid Model and Uncertainty Series ---
ras_TSS_hybridmodel = pd.Series(
    mech_model_output_reindex['.SST_1.X_Under'].values + average_ras_error_preds,
    index=lab_measurements_reindex.index # Use the time index from reindexed data
)

# Create the upper and lower bounds for the uncertainty band
ras_TSS_hybridmodel_upper = pd.Series(
    ras_TSS_hybridmodel.values + std_dev_ras,
    index=lab_measurements_reindex.index # Use same time index
)
ras_TSS_hybridmodel_lower = pd.Series(
    ras_TSS_hybridmodel.values - std_dev_ras,
    index=lab_measurements_reindex.index # Use same time index
)

# --- Plotting ---
plt.figure(figsize=(10,4), dpi=150)
plt.plot(lab_measurements_reindex['TSSras'], '.-', label='Measurements', color='blue')
plt.plot(mech_model_output_reindex['.SST_1.X_Under'], '-', linewidth=1.5, label='Mechanistic model output', color='green') # Commented out as per user code

# Plot hybrid model predictions - use iloc for positional slicing based on plot_split_index
plt.plot(ras_TSS_hybridmodel.iloc[:split_folder_index], '.-', label='HM predictions (train)', color='orange')
plt.plot(ras_TSS_hybridmodel.iloc[split_folder_index:], '.-', label='HM predictions (test)', color='red')

# Plot Standard Deviation Band (Train) - use iloc
plt.fill_between(ras_TSS_hybridmodel.index[:split_folder_index],
                 ras_TSS_hybridmodel_lower.iloc[:split_folder_index],
                 ras_TSS_hybridmodel_upper.iloc[:split_folder_index],
                 color='orange', alpha=0.2, zorder=1)

# Plot Standard Deviation Band (Test) - use iloc
plt.fill_between(ras_TSS_hybridmodel.index[split_folder_index:],
                 ras_TSS_hybridmodel_lower.iloc[split_folder_index:],
                 ras_TSS_hybridmodel_upper.iloc[split_folder_index:],
                 color='red', alpha=0.2, zorder=1)

plt.xlabel("Time")
plt.ylabel("TSS RAS (mg/L)")
plt.title("TSS RAS Prediction using CNN") # Add a title
plt.legend()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path as os_path, listdir
import torch
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
import tensorflow as tf # Import tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

path_to_embeddings = "data/embeddings/autoencoder"
path_to_all_images = 'data/settling_data/filtered_data/images_pileaute_settling_filtered' 
path_to_train_images = 'data/settling_data/filtered_data/train_images_pileaute_settling_filtered'

# Read SVI
SVI=pd.read_csv('data/settling_data/filtered_data/cleaned_SVIs_interpolated.csv', index_col=[0])

# Get Sorted Folders (use actual image path as reference)
train_image_folders=sorted(listdir(path_to_train_images))
all_image_folders = sorted(listdir(path_to_all_images))
num_folders_total = len(all_image_folders)

# --- Load ALL Features, Aggregate, Create Labels and Folder Mapping ---
all_features_list = []
all_labels_SVI_list = []
feature_folder_map = [] # Keep track of which folder index each feature belongs to
processed_folder_indices = [] # Keep track of folders we actually found data for
magnifications_to_process = ["10x", "40x"] # Define which magnifications to use

folder_idx_counter = 0 
for folder_name in train_image_folders:
    # Path to embeddings for this folder
    base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5"
    images_processed_this_folder = 0 # Counter for images successfully loaded from this folder (across mags)

    for mag in magnifications_to_process:
        path_to_embeddings_folder = f"{base_embeddings_path}/{mag}"

        # Check if the specific magnification folder exists and has files
        if not os_path.exists(path_to_embeddings_folder):
            # Don't skip the whole folder yet, maybe the other magnification exists
            print(f"Info: Embeddings path not found for folder {folder_name}, magnification {mag}. Skipping this mag.")
            continue
        if not listdir(path_to_embeddings_folder):
            print(f"Info: Embeddings path empty for folder {folder_name}, magnification {mag}. Skipping this mag.")
            continue

        # Process images within this magnification folder
        images_list_embeddings = listdir(path_to_embeddings_folder)
        for image_file in images_list_embeddings:
            try:
                img_path = os_path.join(path_to_embeddings_folder, image_file)
                embedding = torch.load(img_path).cpu().numpy() # Load embedding

                # --- Feature Aggregation / Flattening ---
                # Example: Flatten the embedding (C, H, W) -> (C*H*W,)
                all_features_list.append(embedding)

                # --- Assign labels corresponding to the PARENT FOLDER ---
                # Both 10x and 40x images from the same 'folder_name' get the same label
                all_labels_SVI_list.append(SVI.loc[folder_name].item())

                # Map this feature vector back to the index in the 'processed_folder_indices' list
                # This index represents the time step / folder group for splitting later
                feature_folder_map.append(len(processed_folder_indices))

                images_processed_this_folder += 1 # Increment count for this folder

            except Exception as e:
                print(f"Error loading or processing {img_path}: {e}")

    if images_processed_this_folder > 0:
        processed_folder_indices.append(folder_idx_counter) # Add original index if processed

    folder_idx_counter += 1 # Move to the next folder's error value

all_features = np.array(all_features_list)
SVI_labels = np.array(all_labels_SVI_list)
feature_folder_map = np.array(feature_folder_map)

print(f"Total folders found: {len(all_image_folders)}")
print(f"Folders successfully processed: {len(processed_folder_indices)}")
print(f"Total image features loaded: {len(all_features)}")
print(f"Labels array length (Effluent): {len(SVI_labels)}")
print(f"Feature matrix shape: {all_features.shape}")
assert len(all_features) == len(SVI_labels) # Check features match labels

# # --- Split Based on Time (Folders) ---
# SPLIT = 0.68
# split_folder_index = round(SPLIT * len(processed_folder_indices))

# # Get indices of features belonging to train folders vs test folders
# train_indices = np.where(feature_folder_map < split_folder_index)[0]
# test_indices = np.where(feature_folder_map >= split_folder_index)[0]

# # --- Create Train/Test Data ---
# X_train = all_features[train_indices]
# y_train = SVI_labels[train_indices]
# X_test = all_features[test_indices]
# y_test = SVI_labels[test_indices]

# print(f"Splitting at processed folder index: {split_folder_index} (out of {len(processed_folder_indices)})")
# print(f"Train set size: {len(train_indices)} images")
# print(f"Test set size: {len(test_indices)} images")

# --- Transpose for Keras ---
# Transpose AFTER splitting
X_train = all_features.transpose(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
#X_test = X_test.transpose(0, 2, 3, 1)
#all_features_transposed = all_features.transpose(0, 2, 3, 1) 

# Check input shape matches X_train
input_shape_keras = X_train.shape[1:] # (H, W, C)
print(f"Input shape for Keras model: {input_shape_keras}")
# Define a function to create the model
def create_model(input_shape):
    model = Sequential()
    
    # First Conv Block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Second Conv Block (deeper)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Third Conv Block (optional, if not too small)
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))  # Explicit linear output
    
    optimizer = Adam(learning_rate=0.0005) 
    model.compile(optimizer=optimizer, loss='mse')
    return model

# --- K-Fold Cross-Validation ---
from sklearn.model_selection import KFold
k = 10  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_histories = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n--- Fold {fold+1} ---")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = SVI_labels[train_idx], SVI_labels[val_idx]
    
    model = create_model(input_shape_keras)
    
    history = model.fit(X_tr, y_tr,
                        epochs=40,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        verbose=1)
    
    fold_histories.append(history)
    fold_models.append(model)



##########################################################################
# testing: on all images 

# --- Load ALL Features, Aggregate, Create Labels and Folder Mapping ---
all_features_list = []
all_labels_SVI_list = []
feature_folder_map = [] # Keep track of which folder index each feature belongs to
processed_folder_indices = [] # Keep track of folders we actually found data for

folder_idx_counter = 0 
for folder_name in all_image_folders:
    # Path to embeddings for this folder
    base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5"
    images_processed_this_folder = 0 # Counter for images successfully loaded from this folder (across mags)

    for mag in magnifications_to_process:
        path_to_embeddings_folder = f"{base_embeddings_path}/{mag}"

        # Check if the specific magnification folder exists and has files
        if not os_path.exists(path_to_embeddings_folder):
            # Don't skip the whole folder yet, maybe the other magnification exists
            print(f"Info: Embeddings path not found for folder {folder_name}, magnification {mag}. Skipping this mag.")
            continue
        if not listdir(path_to_embeddings_folder):
            print(f"Info: Embeddings path empty for folder {folder_name}, magnification {mag}. Skipping this mag.")
            continue

        # Process images within this magnification folder
        images_list_embeddings = listdir(path_to_embeddings_folder)
        for image_file in images_list_embeddings:
            try:
                img_path = os_path.join(path_to_embeddings_folder, image_file)
                embedding = torch.load(img_path).cpu().numpy() # Load embedding

                # --- Feature Aggregation / Flattening ---
                # Example: Flatten the embedding (C, H, W) -> (C*H*W,)
                all_features_list.append(embedding)

                # --- Assign labels corresponding to the PARENT FOLDER ---
                # Both 10x and 40x images from the same 'folder_name' get the same label
                all_labels_SVI_list.append(SVI.loc[folder_name].item())

                # Map this feature vector back to the index in the 'processed_folder_indices' list
                # This index represents the time step / folder group for splitting later
                feature_folder_map.append(len(processed_folder_indices))

                images_processed_this_folder += 1 # Increment count for this folder

            except Exception as e:
                print(f"Error loading or processing {img_path}: {e}")

    if images_processed_this_folder > 0:
        processed_folder_indices.append(folder_idx_counter) # Add original index if processed

    folder_idx_counter += 1 # Move to the next folder's error value


# --- Convert to Numpy Arrays ---
# Feature matrix: rows are images, columns are aggregated features
all_features_agg = np.array(all_features_list) # Shape (num_total_images, num_agg_features)
SVI_labels = np.array(all_labels_SVI_list)
feature_folder_map = np.array(feature_folder_map) 
all_features_transposed = all_features_agg.transpose(0, 2, 3, 1) 
# --- Predict on ALL Data using the model from the last fold ---
y_preds = []

for model in fold_models:
    y_pred = model.predict(all_features_transposed)
    y_preds.append(y_pred)

# Average predictions
y_pred_all = np.mean(y_preds, axis=0) #take average over folds
y_pred_all = y_pred_all.squeeze()

# --- Average Predictions per Time Point (Folder) ---
average_preds = []
std_dev = []
i = 0
for folder_name in all_image_folders:
    print(folder_name)
    # Path to embeddings for this folder
    base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5"
    temporary_pred=[]
    for mag in magnifications_to_process:
        images_list_embeddings = listdir(f"{base_embeddings_path}/{mag}")
        for image_file in images_list_embeddings:
            temporary_pred.append(y_pred_all[i])
            i += 1 
    average_preds.append(np.median(temporary_pred)) 
    std_dev.append(np.std(temporary_pred))

average_preds = np.array(average_preds)
std_dev = np.array(std_dev)
all_image_folders=pd.to_datetime(all_image_folders)
SVI.index=pd.to_datetime(SVI.index)

# --- Construct output and Uncertainty Bands ---
SVI_pred = pd.Series(
    average_preds,
    index=all_image_folders # Use the time index from reindexed data
)

SVI_pred_upper = pd.Series(
    average_preds + std_dev,
    index=all_image_folders
)
SVI_pred_lower = pd.Series(
    average_preds - std_dev,
    index=all_image_folders
)
# --- Plotting ---
# Use the FOLDER-based split index for plotting
plot_split_index = 55 

plt.figure(figsize=(10, 4), dpi=150)
plt.plot(SVI['SVI'], '.-', label='Measurements', color='blue')
plt.plot(SVI_pred.iloc[:plot_split_index], '.-', label='Model predictions (train)', color='orange')
plt.plot(SVI_pred.iloc[plot_split_index:], '.-', label='Model predictions (test)', color='red')

# Plot Standard Deviation Band (Train) - use iloc
plt.fill_between(SVI_pred.index[:plot_split_index],
                 SVI_pred_lower.iloc[:plot_split_index],
                 SVI_pred_upper.iloc[:plot_split_index],
                 color='orange', alpha=0.2, zorder=1)

# Plot Standard Deviation Band (Test) - use iloc
plt.fill_between(SVI_pred.index[plot_split_index:],
                 SVI_pred_lower.iloc[plot_split_index:],
                 SVI_pred_upper.iloc[plot_split_index:],
                 color='red', alpha=0.2, zorder=1)

plt.xlabel("Time")
plt.ylabel("SVI (mL/g)")
plt.title("SVI (autoencoder+CNN)")
plt.legend()
plt.show()
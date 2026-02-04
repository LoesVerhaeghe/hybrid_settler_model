# --- Libraries ---
from utils.helpers import interpolate_time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path as os_path 
from sklearn.ensemble import RandomForestRegressor
import torch
import warnings
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

# --- Load train Features, Aggregate, Create Labels and Folder Mapping ---
all_features_list = []
all_labels_SVI_list = []
feature_folder_map = [] # Keep track of which folder index each feature belongs to
processed_folder_indices = [] # Keep track of folders we actually found data for

folder_idx_counter = 0 
for folder_name in train_image_folders:
    # Path to embeddings for this folder
    path_to_embeddings_folder = f"{path_to_embeddings}/{folder_name}/basin5/10x"

    if not os_path.exists(path_to_embeddings_folder) or not listdir(path_to_embeddings_folder):
        print(f"Warning: Embeddings path not found or empty for folder {folder_name}, skipping.")
        folder_idx_counter += 1 # Still increment error index if skipping folder
        continue

    images_list_embeddings = listdir(path_to_embeddings_folder)
    images_in_folder_count = 0
    for image_file in images_list_embeddings:
        try:
            img_path = f"{path_to_embeddings_folder}/{image_file}"
            embedding = torch.load(img_path).cpu().numpy() # Shape (Channel, Height, Width) e.g. (8, 24, 32)

            # --- Feature Aggregation ---
            # mean/max/min per channel 
            aggregated_features = np.max(embedding, axis=(1, 2)) # Shape: (C,) e.g. (8,)

            # mean/max/min per pixel
            #aggregated_features = np.mean(embedding, axis=0)  

            #aggregated_features= aggregated_features.flatten()

            aggregated_features= embedding.flatten()

            all_features_list.append(aggregated_features)

            # Assign labels corresponding to this FOLDER (time point 'folder_idx_counter')
            all_labels_SVI_list.append(SVI.loc[folder_name].item())
            feature_folder_map.append(len(processed_folder_indices)) # Store the *processed* folder index
            images_in_folder_count += 1

        except Exception as e:
            print(f"Error loading or processing {img_path}: {e}")

    if images_in_folder_count > 0:
        processed_folder_indices.append(folder_idx_counter) # Add original index if processed

    folder_idx_counter += 1 # Move to the next folder's error value

# --- Convert to Numpy Arrays ---
# Feature matrix: rows are images, columns are aggregated features
all_features_agg = np.array(all_features_list) # Shape (num_total_images, num_agg_features)
SVI_labels = np.array(all_labels_SVI_list)
feature_folder_map = np.array(feature_folder_map) 

print(f"Total folders found: {num_folders_total}")
print(f"Folders successfully processed: {len(processed_folder_indices)}")
print(f"Total image features loaded: {len(all_features_agg)}")
print(f"Labels array length (Effluent): {len(SVI_labels)}")
print(f"Feature matrix shape: {all_features_agg.shape}")
assert len(all_features_agg) == len(SVI_labels) # Check features match labels

# # --- Split Based on Time (Processed Folders) ---
# # Use the number of *processed* folders for splitting
# SPLIT = 0.68
# split_folder_index = round(SPLIT * len(processed_folder_indices))

# # Get indices of features belonging to train folders vs test folders based on the processed folder index
# train_indices = np.where(feature_folder_map < split_folder_index)[0]
# test_indices = np.where(feature_folder_map >= split_folder_index)[0]

# print(f"Splitting at processed folder index: {split_folder_index} (out of {len(processed_folder_indices)})")
# print(f"Train set size: {len(train_indices)} images")
# print(f"Test set size: {len(test_indices)} images")

# # --- Create Train/Test Data ---
# X_train = all_features_agg[train_indices]
# y_train = SVI_labels[train_indices]
# X_test = all_features_agg[test_indices]
# y_test = SVI_labels[test_indices]


############################################################
#### Train model for TSS effluent Error (Random Forest) ####
############################################################

# Define the RandomForest model
#the parameters were a little bit optimized (tried 4-5 different pairs)
rf = RandomForestRegressor(n_estimators=50,
                               max_depth=15,
                               random_state=42,
                               n_jobs=-1) # Use all available CPU cores

# Train the model
rf.fit(all_features_agg, SVI_labels)


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
    path_to_embeddings_folder = f"{path_to_embeddings}/{folder_name}/basin5/10x"

    if not os_path.exists(path_to_embeddings_folder) or not listdir(path_to_embeddings_folder):
        print(f"Warning: Embeddings path not found or empty for folder {folder_name}, skipping.")
        folder_idx_counter += 1 # Still increment error index if skipping folder
        continue

    images_list_embeddings = listdir(path_to_embeddings_folder)
    images_in_folder_count = 0
    for image_file in images_list_embeddings:
        try:
            img_path = f"{path_to_embeddings_folder}/{image_file}"
            embedding = torch.load(img_path).cpu().numpy() # Shape (Channel, Height, Width) e.g. (8, 24, 32)

            # --- Feature Aggregation ---
            # mean/max/min per channel 
            #aggregated_features = np.max(embedding, axis=(1, 2)) # Shape: (C,) e.g. (8,)

            # mean/max/min per pixel
            #aggregated_features = np.mean(embedding, axis=0)  

            #aggregated_features= aggregated_features.flatten()

            aggregated_features= embedding.flatten()

            all_features_list.append(aggregated_features)

            # Assign labels corresponding to this FOLDER (time point 'folder_idx_counter')
            all_labels_SVI_list.append(SVI.loc[folder_name].item())
            feature_folder_map.append(len(processed_folder_indices)) # Store the *processed* folder index
            images_in_folder_count += 1

        except Exception as e:
            print(f"Error loading or processing {img_path}: {e}")

    if images_in_folder_count > 0:
        processed_folder_indices.append(folder_idx_counter) # Add original index if processed

    folder_idx_counter += 1 # Move to the next folder's error value

# --- Convert to Numpy Arrays ---
# Feature matrix: rows are images, columns are aggregated features
all_features_agg = np.array(all_features_list) # Shape (num_total_images, num_agg_features)
SVI_labels = np.array(all_labels_SVI_list)
feature_folder_map = np.array(feature_folder_map) 


# --- Predict on ALL Data for Hybrid Model ---
y_pred_all = rf.predict(all_features_agg)

# --- Average Predictions and Calculate Std Dev per Time Point (Folder) ---
average_preds = []
std_dev = []
i = 0
for folder_name in all_image_folders: 
    base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5/10x"
    temporary_pred=[]
    images_list_embeddings = listdir(base_embeddings_path)
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
    SVI_pred.values + std_dev,
    index=all_image_folders
)
SVI_pred_lower = pd.Series(
    SVI_pred.values - std_dev,
    index=all_image_folders
)

# Use the FOLDER-based split index for plotting consistency
plot_split_index = 55 

plt.figure(figsize=(13, 4), dpi=150)
plt.plot(SVI, '.-', label='Measurements', color='blue')
plt.plot(SVI_pred.iloc[:plot_split_index], '.-', label='HM predictions (train)', color='orange')
plt.plot(SVI_pred.iloc[plot_split_index:], '.-', label='HM predictions (test)', color='red')

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
plt.title("SVI (RF)")
plt.legend()
plt.show()


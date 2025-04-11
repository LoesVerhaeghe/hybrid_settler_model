# --- Libraries ---
from utils.helpers import interpolate_time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path as os_path 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score 
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# --- Configuration & Data Reading ---
start_date = pd.to_datetime("2023-10-01")
SPLIT = 0.65
path_to_embeddings = "data/embeddings/autoencoder"
path_to_all_images = 'data/microscope_images' 

# Read SVI
SVI=pd.read_csv('data/SVI.csv', index_col=[0])

# Get Sorted Folders (use actual image path as reference)
all_image_folders = sorted(listdir(path_to_all_images))
num_folders_total = len(all_image_folders)

# Interpolate Measurements/Model Output to Folder Dates
SVI_reindex = interpolate_time(SVI, all_image_folders)
SVI=SVI_reindex.values

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
            #aggregated_features = np.min(embedding, axis=(1, 2)) # Shape: (C,) e.g. (8,)

            # mean/max/min per pixel
            #aggregated_features = np.min(embedding, axis=0)  

            #aggregated_features= aggregated_features.flatten()

            aggregated_features= embedding.flatten()

            all_features_list.append(aggregated_features)

            # Assign labels corresponding to this FOLDER (time point 'folder_idx_counter')
            all_labels_SVI_list.append(SVI[folder_idx_counter])
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

# --- Split Based on Time (Processed Folders) ---
# Use the number of *processed* folders for splitting
split_folder_index = round(SPLIT * len(processed_folder_indices))

# Get indices of features belonging to train folders vs test folders based on the processed folder index
train_indices = np.where(feature_folder_map < split_folder_index)[0]
test_indices = np.where(feature_folder_map >= split_folder_index)[0]

print(f"Splitting at processed folder index: {split_folder_index} (out of {len(processed_folder_indices)})")
print(f"Train set size: {len(train_indices)} images")
print(f"Test set size: {len(test_indices)} images")

# --- Create Train/Test Data ---
X_train = all_features_agg[train_indices]
y_train = SVI_labels[train_indices]
X_test = all_features_agg[test_indices]
y_test = SVI_labels[test_indices]

# # --- Feature processing with PCA (Fit on Train, Transform Train/Test) ---
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# all_features_agg_scaled = scaler.transform(all_features_agg)

# pca = PCA(n_components=15, random_state=42)
# pca.fit(X_train_scaled)
# X_train = pca.transform(X_train_scaled)
# X_test = pca.transform(X_test_scaled)
# all_features_agg = pca.transform(all_features_agg_scaled)

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
rf.fit(X_train, y_train)

# --- Evaluate on Explicit Test Set ---
y_pred_test = rf.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print('Evaluation metric:')
print(f"Test MAE (Effluent Error): {mae_test:.4f}")
print(f"Test R2 Score (Effluent Error): {r2_test:.4f}")

# --- Predict on ALL Data for Hybrid Model ---
y_pred_all = rf.predict(all_features_agg)

# --- Average Predictions and Calculate Std Dev per Time Point (Folder) ---
average_eff_error_preds = []
std_dev_eff = []
current_pred_index = 0
for folder_idx in range(len(processed_folder_indices)): # Iterate through processed folders
    # Find how many features belong to this folder (using the processed folder index)
    num_images_in_folder = np.sum(feature_folder_map == folder_idx)

    if num_images_in_folder > 0:
        preds_for_folder = y_pred_all[current_pred_index : current_pred_index + num_images_in_folder]
        average_eff_error_preds.append(np.median(preds_for_folder)) 
        std_dev_eff.append(np.std(preds_for_folder))
        current_pred_index += num_images_in_folder
    else:
        print(f"Warning: No predictions found for processed folder index {folder_idx}. Appending NaN.")
        average_eff_error_preds.append(np.nan)
        std_dev_eff.append(np.nan)

average_eff_error_preds = np.array(average_eff_error_preds)
std_dev_eff = np.array(std_dev_eff)

# --- Construct Hybrid Effluent Model and Uncertainty Bands ---
SVI_pred = pd.Series(
    average_eff_error_preds,
    index=SVI_reindex.index # Use the time index from reindexed data
)

SVI_pred_upper = pd.Series(
    SVI_pred.values + std_dev_eff,
    index=SVI_reindex.index
)
SVI_pred_lower = pd.Series(
    SVI_pred.values - std_dev_eff,
    index=SVI_reindex.index
)


# --- Plotting Effluent Results ---
# Use the FOLDER-based split index for plotting consistency
plot_split_index = split_folder_index 

plt.figure(figsize=(10, 4), dpi=150)
plt.plot(SVI_reindex['SVI'], '.-', label='Measurements', color='blue')
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


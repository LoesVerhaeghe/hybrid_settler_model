import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path as os_path 
from sklearn.ensemble import RandomForestRegressor
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

path_to_embeddings = "data/embeddings/autoencoder"
path_to_all_images = 'data/beluchting/images_pileaute_kla' 

all_image_folders = sorted(listdir(path_to_all_images))
num_folders_total = len(all_image_folders)

df_kla=pd.read_csv('data/beluchting/kla_values.csv', index_col=0)
df_Qair_residuals=pd.read_csv('data/beluchting/Qair_residuals_interpolated.csv', index_col=0)

# --- Load ALL Features, Aggregate, Create Labels and Folder Mapping ---
all_features_list = []
all_labels_kla_list = []
all_labels_Qairres_list = []
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
            #aggregated_features = np.mean(embedding, axis=(1, 2)) # Shape: (C,) e.g. (8,)

            # mean/max/min per pixel
            #embedding = np.min(embedding, axis=0)  
            aggregated_features= embedding.flatten()

            all_features_list.append(aggregated_features)

            # Assign labels corresponding to this FOLDER (time point 'folder_idx_counter')
            all_labels_kla_list.append(df_kla['kLa'].loc[folder_name].item())
            all_labels_Qairres_list.append(df_Qair_residuals['airflow_residueel'].loc[folder_name].item())
            feature_folder_map.append(len(processed_folder_indices)) # Store the *processed* folder index
            images_in_folder_count += 1

        except Exception as e:
            print(f"Error loading or processing {img_path}: {e}")

    if images_in_folder_count > 0:
        processed_folder_indices.append(folder_idx_counter) # Add original index if processed

    folder_idx_counter += 1 # Move to the next folder's error value

all_features_agg = np.array(all_features_list) # Shape (num_total_images, num_agg_features)
kla_labels = np.array(all_labels_kla_list)
Qairres_labels = np.array(all_labels_Qairres_list)
feature_folder_map = np.array(feature_folder_map) 

print(f"Total folders found: {num_folders_total}")
print(f"Folders successfully processed: {len(processed_folder_indices)}")
print(f"Total image features loaded: {len(all_features_agg)}")
print(f"Labels array length: {len(kla_labels)}")
print(f"Feature matrix shape: {all_features_agg.shape}")
assert len(all_features_agg) == len(kla_labels) # Check features match labels

# --- Split Based on Time (Processed Folders) ---
train_range1 = np.arange(0, 30)
train_range2 = np.arange(59, 87)
train_indices_folders = np.concatenate((train_range1, train_range2))
test_indices_folders=np.arange(30, 59)

# Get indices of features belonging to train folders vs test folders
train_indices = np.where(np.isin(feature_folder_map, train_indices_folders))[0]
test_indices = np.where(np.isin(feature_folder_map, test_indices_folders))[0]

print(f"Train set size: {len(train_indices)} images")
print(f"Test set size: {len(test_indices)} images")

# --- Create Train/Test Data ---
X_train = all_features_agg[train_indices]
y_train_kla = kla_labels[train_indices]
X_test = all_features_agg[test_indices]
y_test_kla = kla_labels[test_indices]

y_train_ras = Qairres_labels[train_indices]
y_test_ras = Qairres_labels[test_indices]


############################################################
#### Train model (Random Forest) ####
############################################################

# Define the RandomForest model
#the parameters were a little bit optimized (tried 4-5 different pairs)
rf_kla = RandomForestRegressor(n_estimators=15,
                               max_depth=10,
                               random_state=42,
                               n_jobs=-1) # Use all available CPU cores

# Train the model
rf_kla.fit(X_train, y_train_kla)

# --- Predict on ALL Data for Hybrid Model ---
y_pred_kla_all = rf_kla.predict(all_features_agg)

# --- Average Predictions and Calculate Std Dev per Time Point (Folder) ---
average_kla_preds = []
std_dev_kla = []
i=0
for folder_name in all_image_folders:
    # Path to embeddings for this folder
    base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5/10x"
    temporary_pred=[]
    images_list_embeddings = listdir(base_embeddings_path)
    for image_file in images_list_embeddings:
        temporary_pred.append(y_pred_kla_all[i])
        i += 1 
    average_kla_preds.append(np.median(temporary_pred)) 
    std_dev_kla.append(np.std(temporary_pred))

all_image_folders_datetime=pd.to_datetime(all_image_folders)
df_kla.index=pd.to_datetime(df_kla.index)

# --- Construct Model preds and Uncertainty Bands ---
# Ensure index alignment - crucial if folders were skipped
kla_predictions = pd.Series(
    average_kla_preds,
    index=all_image_folders_datetime
)

kla_upper = pd.Series(
    kla_predictions.values + std_dev_kla,
    index=all_image_folders_datetime
)
kla_lower = pd.Series(
    kla_predictions.values - std_dev_kla,
    index=all_image_folders_datetime
)

# --- Plotting ---
# Use the FOLDER-based split index for plotting
plt.figure(figsize=(10, 4), dpi=150)
plt.plot(df_kla['kLa'], '.-', label='calculated KLa', color='blue')
plt.plot(kla_predictions.iloc[train_range1], '.-', label='Model predictions (train)', color='orange')
plt.plot(kla_predictions.iloc[train_range2], '.-', color='orange')
plt.plot(kla_predictions.iloc[test_indices_folders], '.-', label='Model predictions (test)', color='red')
plt.fill_between(kla_predictions.index[train_range1],
                 kla_lower[train_range1],
                 kla_upper[train_range1],
                 color='orange', alpha=0.2, zorder=1)
plt.fill_between(kla_predictions.index[train_range2],
                 kla_lower[train_range2],
                 kla_upper[train_range2],
                 color='orange', alpha=0.2, zorder=1)
plt.fill_between(kla_predictions.index[test_indices_folders],
                 kla_lower[test_indices_folders],
                 kla_upper[test_indices_folders],
                 color='red', alpha=0.2, zorder=1)
plt.xlabel("Time")
plt.ylabel("KLa")
plt.legend()
plt.show()

########################################################
#### Train model (Random Forest) ####
########################################################

# Define the second RandomForest model
#the parameters were a little bit optimized (tried 4-5 different pairs)
rf_Qairres = RandomForestRegressor(n_estimators=15, 
                               max_depth=10,     
                               random_state=42,
                               n_jobs=-1) # Use all available CPU cores

# Train the model (using the same X_train but different labels y_train_ras)
rf_Qairres.fit(X_train, y_train_ras)

y_pred_Qairres = rf_Qairres.predict(all_features_agg)
all_image_folders = sorted(listdir(path_to_all_images))
# --- Average Predictions and Calculate Std Dev per Time Point (Folder) ---
average_Qairres_preds = []
std_dev_Qairres = []
i = 0
for folder_name in all_image_folders: # Iterate through processed folders
    # Path to embeddings for this folder
    base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5/10x"
    temporary_pred=[]
    images_list_embeddings = listdir(base_embeddings_path)
    for image_file in images_list_embeddings:
        temporary_pred.append(y_pred_Qairres[i])
        i += 1 
    average_Qairres_preds.append(np.median(temporary_pred)) 
    std_dev_Qairres.append(np.std(temporary_pred))

df_Qair_residuals.index=pd.to_datetime(df_Qair_residuals.index)

# --- Construct Hybrid RAS Model and Uncertainty Bands ---
Qairres_preds = pd.Series(
    average_Qairres_preds,
    index=all_image_folders_datetime # Use the time index from reindexed data
)

Qairres_upper = pd.Series(
    Qairres_preds + std_dev_Qairres,
    index=all_image_folders_datetime
)
Qairres_lower = pd.Series(
    Qairres_preds - std_dev_Qairres,
    index=all_image_folders_datetime
)

# --- Plotting ---
plt.figure(figsize=(10,4), dpi=150)
plt.plot(df_Qair_residuals['airflow_residueel'], '.-', label='Measurements', color='blue')
plt.plot(Qairres_preds.iloc[train_range1], '.-', label='Model predictions (train)', color='orange')
plt.plot(Qairres_preds.iloc[train_range2], '.-', color='orange')
plt.plot(Qairres_preds.iloc[test_indices_folders], '.-', label='Model predictions (test)', color='red')
plt.fill_between(Qairres_preds.index[train_range1],
                 Qairres_lower.iloc[train_range1],
                 Qairres_upper.iloc[train_range1],
                 color='orange', alpha=0.2, zorder=1)
plt.fill_between(Qairres_preds.index[test_indices_folders],
                 Qairres_lower.iloc[test_indices_folders],
                 Qairres_upper.iloc[test_indices_folders],
                 color='red', alpha=0.2, zorder=1)
plt.fill_between(Qairres_preds.index[train_range2],
                 Qairres_lower[train_range2],
                 Qairres_upper[train_range2],
                 color='orange', alpha=0.2, zorder=1)
plt.xlabel("Time")
plt.ylabel("Qair residuals")
plt.legend()
plt.show()


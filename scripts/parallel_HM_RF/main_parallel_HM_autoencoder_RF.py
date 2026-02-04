import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path as os_path 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score 
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

path_to_embeddings = "data/embeddings/autoencoder"
path_to_all_images = 'data/settling_data/filtered_data/images_pileaute_settling_filtered' 

all_image_folders = sorted(listdir(path_to_all_images))
num_folders_total = len(all_image_folders)

df_TSS_eff=pd.read_excel('data/settling_data/filtered_data/filtered_TSS_data.xlsx', index_col=0, sheet_name='TSS_effluent')
df_TSS_ras=pd.read_excel('data/settling_data/filtered_data/filtered_TSS_data.xlsx', index_col=0, sheet_name='TSS_RAS')

# --- Load ALL Features, Aggregate, Create Labels and Folder Mapping ---
all_features_list = []
all_labels_eff_list = []
all_labels_ras_list = []
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
            #embedding = np.mean(embedding, axis=0)  
            aggregated_features= embedding.flatten()

            all_features_list.append(aggregated_features)

            # Assign labels corresponding to this FOLDER (time point 'folder_idx_counter')
            all_labels_eff_list.append(df_TSS_eff['Error'].loc[folder_name].item())
            all_labels_ras_list.append(df_TSS_ras['WeeklyError'].loc[folder_name].item())
            feature_folder_map.append(len(processed_folder_indices)) # Store the *processed* folder index
            images_in_folder_count += 1

        except Exception as e:
            print(f"Error loading or processing {img_path}: {e}")

    if images_in_folder_count > 0:
        processed_folder_indices.append(folder_idx_counter) # Add original index if processed

    folder_idx_counter += 1 # Move to the next folder's error value

all_features_agg = np.array(all_features_list) # Shape (num_total_images, num_agg_features)
TSSeffluent_labels = np.array(all_labels_eff_list)
TSSras_labels = np.array(all_labels_ras_list)
feature_folder_map = np.array(feature_folder_map) 

print(f"Total folders found: {num_folders_total}")
print(f"Folders successfully processed: {len(processed_folder_indices)}")
print(f"Total image features loaded: {len(all_features_agg)}")
print(f"Labels array length (Effluent): {len(TSSeffluent_labels)}")
print(f"Feature matrix shape: {all_features_agg.shape}")
assert len(all_features_agg) == len(TSSeffluent_labels) # Check features match labels

# --- Split Based on Time (Processed Folders) ---
# Use the number of *processed* folders for splitting
train_indices_folders=np.arange(0, 55)
test_indices_folders=np.arange(55, 81)

# Get indices of features belonging to train folders vs test folders
train_indices = np.where(np.isin(feature_folder_map, train_indices_folders))[0]
test_indices = np.where(np.isin(feature_folder_map, test_indices_folders))[0]

print(f"Train set size: {len(train_indices)} images")
print(f"Test set size: {len(test_indices)} images")

# --- Create Train/Test Data ---
X_train = all_features_agg[train_indices]
y_train_eff = TSSeffluent_labels[train_indices]
X_test = all_features_agg[test_indices]
y_test_eff = TSSeffluent_labels[test_indices]

y_train_ras = TSSras_labels[train_indices]
y_test_ras = TSSras_labels[test_indices]

# --- Feature processing with PCA (Fit on Train, Transform Train/Test) ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
rf_eff = RandomForestRegressor(n_estimators=50,
                               max_depth=15,
                               random_state=42,
                               n_jobs=-1) # Use all available CPU cores

# Train the model
rf_eff.fit(X_train, y_train_eff)

# --- Evaluate on Explicit Test Set ---
y_pred_test_eff = rf_eff.predict(X_test)
mae_test_eff = mean_absolute_error(y_test_eff, y_pred_test_eff)
r2_test_eff = r2_score(y_test_eff, y_pred_test_eff)
print('Evaluation metric:')
print(f"Test MAE (Effluent Error): {mae_test_eff:.4f}")
print(f"Test R2 Score (Effluent Error): {r2_test_eff:.4f}")

# --- Predict on ALL Data for Hybrid Model ---
y_pred_eff_all = rf_eff.predict(all_features_agg)

# --- Average Predictions and Calculate Std Dev per Time Point (Folder) ---
average_eff_error_preds = []
std_dev_eff = []
i=0
for folder_name in all_image_folders:
    # Path to embeddings for this folder
    base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5/10x"
    temporary_pred=[]
    images_list_embeddings = listdir(base_embeddings_path)
    for image_file in images_list_embeddings:
        temporary_pred.append(y_pred_eff_all[i])
        i += 1 
    average_eff_error_preds.append(np.median(temporary_pred)) 
    std_dev_eff.append(np.std(temporary_pred))

average_eff_error_preds = np.array(average_eff_error_preds)
std_dev_eff = np.array(std_dev_eff)
all_image_folders=pd.to_datetime(all_image_folders)
df_TSS_eff.index=pd.to_datetime(df_TSS_eff.index)

# --- Construct Hybrid Effluent Model and Uncertainty Bands ---
# Ensure index alignment - crucial if folders were skipped
eff_TSS_hybridmodel = pd.Series(
    average_eff_error_preds+df_TSS_eff['MechModelOutput'],
    index=all_image_folders
)

eff_TSS_hybridmodel_upper = pd.Series(
    eff_TSS_hybridmodel.values + std_dev_eff,
    index=all_image_folders
)
eff_TSS_hybridmodel_lower = pd.Series(
    eff_TSS_hybridmodel.values - std_dev_eff,
    index=all_image_folders
)

# --- Plotting Effluent Results ---


plt.figure(figsize=(14, 3), dpi=200)
plt.rcParams.update({'font.size': 12})    

plt.plot(df_TSS_eff['LabMeasurement'], '.-', label='Measurements', color='blue')
plt.plot(df_TSS_eff['MechModelOutput'], '.-', label='Mechanistic model output', color='green')

# Plot hybrid model predictions - use iloc for positional slicing based on plot_split_index
plt.plot(eff_TSS_hybridmodel.iloc[train_indices_folders], '.-', label='HM predictions (train)', color='orange')
plt.plot(eff_TSS_hybridmodel.iloc[test_indices_folders], '.-', label='HM predictions (test)', color='red')

# Plot Standard Deviation Band (Train) - use iloc
plt.fill_between(eff_TSS_hybridmodel.index[train_indices_folders],
                 eff_TSS_hybridmodel_lower.iloc[train_indices_folders],
                 eff_TSS_hybridmodel_upper.iloc[train_indices_folders],
                 color='orange', alpha=0.2, zorder=1)

# Plot Standard Deviation Band (Test) - use iloc
plt.fill_between(eff_TSS_hybridmodel.index[test_indices_folders],
                 eff_TSS_hybridmodel_lower.iloc[test_indices_folders],
                 eff_TSS_hybridmodel_upper.iloc[test_indices_folders],
                 color='red', alpha=0.2, zorder=1)

plt.xlabel("Time")
plt.ylabel("TSS effluent (mg/L)")
plt.legend()
plt.show()

########################################################
#### Train model for TSS RAS Error (Random Forest) ####
########################################################

# # Define the second RandomForest model
# #the parameters were a little bit optimized (tried 4-5 different pairs)
# rf_ras = RandomForestRegressor(n_estimators=50, 
#                                max_depth=15,     
#                                random_state=42,
#                                n_jobs=-1) # Use all available CPU cores

# # Train the model (using the same X_train but different labels y_train_ras)
# rf_ras.fit(X_train, y_train_ras)

# # --- Evaluate on Explicit Test Set ---
# y_pred_ras = rf_ras.predict(all_features_agg)
# all_image_folders = sorted(listdir(path_to_all_images))
# # --- Average Predictions and Calculate Std Dev per Time Point (Folder) ---
# average_ras_error_preds = []
# std_dev_ras = []
# i = 0
# for folder_name in all_image_folders: # Iterate through processed folders
#     # Path to embeddings for this folder
#     base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5/10x"
#     temporary_pred=[]
#     images_list_embeddings = listdir(base_embeddings_path)
#     for image_file in images_list_embeddings:
#         temporary_pred.append(y_pred_ras[i])
#         i += 1 
#     average_ras_error_preds.append(np.median(temporary_pred)) 
#     std_dev_ras.append(np.std(temporary_pred))

# average_ras_error_preds = np.array(average_ras_error_preds)
# std_dev_ras = np.array(std_dev_ras)
# all_image_folders=pd.to_datetime(all_image_folders)
# df_TSS_ras.index=pd.to_datetime(df_TSS_ras.index)

# # --- Construct Hybrid RAS Model and Uncertainty Bands ---
# ras_TSS_hybridmodel = pd.Series(
#     average_ras_error_preds+df_TSS_ras['WeeklyMechModelOutput'],
#     index=all_image_folders # Use the time index from reindexed data
# )

# ras_TSS_hybridmodel_upper = pd.Series(
#     ras_TSS_hybridmodel + std_dev_ras,
#     index=all_image_folders
# )
# ras_TSS_hybridmodel_lower = pd.Series(
#     ras_TSS_hybridmodel - std_dev_ras,
#     index=all_image_folders
# )


# # --- Plotting ---
# plt.figure(figsize=(10,4), dpi=150)
# plt.plot(df_TSS_ras['LabMeasurement'], '.-', label='Measurements', color='blue')
# plt.plot(df_TSS_ras['WeeklyMechModelOutput'], '-', linewidth=1.5, label='Mechanistic model output', color='green') # Commented out as per user code

# # Plot hybrid model predictions - use iloc for positional slicing based on plot_split_index
# plt.plot(ras_TSS_hybridmodel.iloc[train_indices_folders], '.-', label='HM predictions (train)', color='orange')
# plt.plot(ras_TSS_hybridmodel.iloc[test_indices_folders], '.-', label='HM predictions (test)', color='red')

# # Plot Standard Deviation Band (Train) - use iloc
# plt.fill_between(ras_TSS_hybridmodel.index[train_indices_folders],
#                  ras_TSS_hybridmodel_lower.iloc[train_indices_folders],
#                  ras_TSS_hybridmodel_upper.iloc[train_indices_folders],
#                  color='orange', alpha=0.2, zorder=1)

# # Plot Standard Deviation Band (Test) - use iloc
# plt.fill_between(ras_TSS_hybridmodel.index[test_indices_folders],
#                  ras_TSS_hybridmodel_lower.iloc[test_indices_folders],
#                  ras_TSS_hybridmodel_upper.iloc[test_indices_folders],
#                  color='red', alpha=0.2, zorder=1)

# plt.xlabel("Time")
# plt.ylabel("TSS RAS (mg/L)")
# plt.legend()
# plt.show()
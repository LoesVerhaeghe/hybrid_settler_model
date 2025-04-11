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
            #aggregated_features = np.mean(embedding, axis=(1, 2)) # Shape: (C,) e.g. (8,)

            # mean/max/min per pixel
            #embedding = np.mean(embedding, axis=0)  
            aggregated_features= embedding.flatten()

            all_features_list.append(aggregated_features)

            # Assign labels corresponding to this FOLDER (time point 'folder_idx_counter')
            all_labels_eff_list.append(TSS_eff_error[folder_idx_counter])
            all_labels_ras_list.append(TSS_ras_error[folder_idx_counter])
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
TSSeffluent_labels = np.array(all_labels_eff_list)
TSSras_labels = np.array(all_labels_ras_list)
feature_folder_map = np.array(feature_folder_map) 

# Adjust reindexed dataframes to only include processed folders
lab_measurements_reindex = lab_measurements_reindex.iloc[processed_folder_indices]
mech_model_output_reindex = mech_model_output_reindex.iloc[processed_folder_indices]
TSS_eff_error = TSS_eff_error[processed_folder_indices] # Keep only errors for processed folders
TSS_ras_error = TSS_ras_error[processed_folder_indices] # Keep only errors for processed folders

print(f"Total folders found: {num_folders_total}")
print(f"Folders successfully processed: {len(processed_folder_indices)}")
print(f"Total image features loaded: {len(all_features_agg)}")
print(f"Labels array length (Effluent): {len(TSSeffluent_labels)}")
print(f"Feature matrix shape: {all_features_agg.shape}")
assert len(all_features_agg) == len(TSSeffluent_labels) # Check features match labels
assert len(lab_measurements_reindex) == len(processed_folder_indices) # Check reindexed length

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
y_train_eff = TSSeffluent_labels[train_indices]
X_test = all_features_agg[test_indices]
y_test_eff = TSSeffluent_labels[test_indices]

y_train_ras = TSSras_labels[train_indices]
y_test_ras = TSSras_labels[test_indices]



# --- Feature processing with PCA (Fit on Train, Transform Train/Test) ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
all_features_agg_scaled = scaler.transform(all_features_agg)

pca = PCA(n_components=15, random_state=42)
pca.fit(X_train_scaled)
X_train = pca.transform(X_train_scaled)
X_test = pca.transform(X_test_scaled)
all_features_agg = pca.transform(all_features_agg_scaled)

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
current_pred_index = 0
for folder_idx in range(len(processed_folder_indices)): # Iterate through processed folders
    # Find how many features belong to this folder (using the processed folder index)
    num_images_in_folder = np.sum(feature_folder_map == folder_idx)

    if num_images_in_folder > 0:
        preds_for_folder = y_pred_eff_all[current_pred_index : current_pred_index + num_images_in_folder]
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
eff_TSS_hybridmodel = pd.Series(
    mech_model_output_reindex['.SST_1.X_Out'].values + average_eff_error_preds,
    index=lab_measurements_reindex.index # Use the time index from reindexed data
)

eff_TSS_hybridmodel_upper = pd.Series(
    eff_TSS_hybridmodel.values + std_dev_eff,
    index=lab_measurements_reindex.index
)
eff_TSS_hybridmodel_lower = pd.Series(
    eff_TSS_hybridmodel.values - std_dev_eff,
    index=lab_measurements_reindex.index
)


# --- Plotting Effluent Results ---
# Use the FOLDER-based split index for plotting consistency
plot_split_index = split_folder_index 

plt.figure(figsize=(10, 4), dpi=150)
plt.plot(lab_measurements_reindex['TSSeffluent'], '.-', label='Measurements', color='blue')
plt.plot(mech_model_output_reindex['.SST_1.X_Out'], '-', linewidth=1.5, label='Mechanistic model output', color='green')

# Plot hybrid model predictions - use iloc for positional slicing
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
plt.title("TSS Effluent Hybrid Model (Mechanistic + RF Error Correction)")
plt.legend()
plt.show()


########################################################
#### Train model for TSS RAS Error (Random Forest) ####
########################################################

# Define the second RandomForest model
#the parameters were a little bit optimized (tried 4-5 different pairs)
rf_ras = RandomForestRegressor(n_estimators=50, 
                               max_depth=15,     
                               random_state=42,
                               n_jobs=-1) # Use all available CPU cores

# Train the model (using the same X_train but different labels y_train_ras)
rf_ras.fit(X_train, y_train_ras)

# --- Evaluate on Explicit Test Set ---
y_pred_test_ras = rf_ras.predict(X_test)
mae_test_ras = mean_absolute_error(y_test_ras, y_pred_test_ras)
r2_test_ras = r2_score(y_test_ras, y_pred_test_ras)
print(f"Test MAE (RAS Error): {mae_test_ras:.4f}")
print(f"Test R2 Score (RAS Error): {r2_test_ras:.4f}")

# --- Predict on ALL Data for Hybrid Model ---
y_pred_ras_all = rf_ras.predict(all_features_agg)

# --- Average Predictions and Calculate Std Dev per Time Point (Folder) ---
average_ras_error_preds = []
std_dev_ras = []
current_pred_index = 0
for folder_idx in range(len(processed_folder_indices)): # Iterate through processed folders
    # Find how many features belong to this folder
    num_images_in_folder = np.sum(feature_folder_map == folder_idx)

    if num_images_in_folder > 0:
        preds_for_folder = y_pred_ras_all[current_pred_index : current_pred_index + num_images_in_folder]
        average_ras_error_preds.append(np.median(preds_for_folder))
        std_dev_ras.append(np.std(preds_for_folder))
        current_pred_index += num_images_in_folder
    else:
        print(f"Warning: No RAS predictions found for processed folder index {folder_idx}. Appending NaN.")
        average_ras_error_preds.append(np.nan)
        std_dev_ras.append(np.nan)

average_ras_error_preds = np.array(average_ras_error_preds)
std_dev_ras = np.array(std_dev_ras)

# --- Construct Hybrid RAS Model and Uncertainty Bands ---
ras_TSS_hybridmodel = pd.Series(
    mech_model_output_reindex['.SST_1.X_Under'].values + average_ras_error_preds,
    index=lab_measurements_reindex.index # Use the time index from reindexed data
)

ras_TSS_hybridmodel_upper = pd.Series(
    ras_TSS_hybridmodel.values + std_dev_ras,
    index=lab_measurements_reindex.index
)
ras_TSS_hybridmodel_lower = pd.Series(
    ras_TSS_hybridmodel.values - std_dev_ras,
    index=lab_measurements_reindex.index
)

# --- Plotting RAS Results ---
plt.figure(figsize=(10, 4), dpi=150)
plt.plot(lab_measurements_reindex['TSSras'], '.-', label='Measurements', color='blue')
plt.plot(mech_model_output_reindex['.SST_1.X_Under'], '-', linewidth=1.5, label='Mechanistic model output', color='green')

# Plot hybrid model predictions - use iloc for positional slicing
plt.plot(ras_TSS_hybridmodel.iloc[:plot_split_index], '.-', label='HM predictions (train)', color='orange')
plt.plot(ras_TSS_hybridmodel.iloc[plot_split_index:], '.-', label='HM predictions (test)', color='red')

# Plot Standard Deviation Band (Train) - use iloc
plt.fill_between(ras_TSS_hybridmodel.index[:plot_split_index],
                 ras_TSS_hybridmodel_lower.iloc[:plot_split_index],
                 ras_TSS_hybridmodel_upper.iloc[:plot_split_index],
                 color='orange', alpha=0.2, zorder=1)

# Plot Standard Deviation Band (Test) - use iloc
plt.fill_between(ras_TSS_hybridmodel.index[plot_split_index:],
                 ras_TSS_hybridmodel_lower.iloc[plot_split_index:],
                 ras_TSS_hybridmodel_upper.iloc[plot_split_index:],
                 color='red', alpha=0.2, zorder=1)

plt.xlabel("Time")
plt.ylabel("TSS RAS (mg/L)")
plt.title("TSS RAS Hybrid Model (Mechanistic + RF Error Correction)")
plt.legend()
plt.show()
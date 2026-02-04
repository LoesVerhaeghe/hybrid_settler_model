import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path as os_path, listdir
import torch
import warnings
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

path_to_embeddings = "data/embeddings/microscope_images_encoded_jointloss"
path_to_all_images = 'data/settling_data/filtered_data/images_pileaute_settling_filtered' 

all_image_folders = sorted(listdir(path_to_all_images))
num_folders_total = len(all_image_folders)

df_TSS_eff=pd.read_excel('data/settling_data/filtered_data/filtered_TSS_data.xlsx', index_col=0, sheet_name='TSS_effluent')
df_TSS_ras=pd.read_excel('data/settling_data/filtered_data/filtered_TSS_data.xlsx', index_col=0, sheet_name='TSS_RAS')

# --- Load ALL Features and Create Corresponding Labels and Folder Mapping ---
all_features_list = []
all_labels_eff_list = []
all_labels_ras_list = []
feature_folder_map = [] # Keep track of which folder each feature belongs to
processed_folder_indices = [] # Keep track of folders we actually found data for
magnifications_to_process = ["10x", "40x"] # Define which magnifications to use

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
                all_labels_eff_list.append(df_TSS_eff['LabMeasurement'].loc[folder_name].item())
                all_labels_ras_list.append(df_TSS_ras['LabMeasurement'].loc[folder_name].item())

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
TSSeffluent_labels = np.array(all_labels_eff_list)
TSSras_labels = np.array(all_labels_ras_list) 
feature_folder_map = np.array(feature_folder_map)

print(f"Total folders found: {len(all_image_folders)}")
print(f"Folders successfully processed: {len(processed_folder_indices)}")
print(f"Total image features loaded: {len(all_features)}")
print(f"Labels array length (Effluent): {len(TSSeffluent_labels)}")
print(f"Feature matrix shape: {all_features.shape}")
assert len(all_features) == len(TSSeffluent_labels) # Check features match labels

# --- Split Based on Time (Folders) ---
train_range1 = np.arange(0, 27)
train_range2 = np.arange(54, 81)
train_indices_folders = np.concatenate((train_range1, train_range2))
test_indices_folders=np.arange(27, 54)

# Get indices of features belonging to train folders vs test folders
train_indices = np.where(np.isin(feature_folder_map, train_indices_folders))[0]
test_indices = np.where(np.isin(feature_folder_map, test_indices_folders))[0]

# Create train/test sets
X_train = all_features[train_indices]
y_train_eff = TSSeffluent_labels[train_indices]
X_test = all_features[test_indices]
y_test_eff = TSSeffluent_labels[test_indices]

y_train_ras = TSSras_labels[train_indices]
y_test_ras = TSSras_labels[test_indices]

print(f"Train set size: {len(train_indices)} images")
print(f"Test set size: {len(test_indices)} images")

# Flatten inputs
X_flat = X_train.reshape((X_train.shape[0], -1))  # (N, 32*24*8)
all_features_flat= all_features.reshape((all_features.shape[0], -1))
y = y_train_eff  # shape (N,)

# Set number of folds
n_splits = 4
kf = KFold(n_splits=n_splits, shuffle=True, random_state=69)

# Store scores
mse_scores = []
mae_scores = []
models=[]
all_predictions=[]
# Start cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_flat)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")
    
    X_tr, X_val = X_flat[train_idx], X_flat[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Define model
    model = Sequential([
        Input(shape=(X_flat.shape[1],)),
        Dense(512, activation='relu'),
        Dropout(0.1),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_tr, y_tr,
              epochs=10000,
              batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=[early_stop],
              verbose=0)

    # Predict and evaluate
    y_val_pred = model.predict(X_val).squeeze()
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)

    mse_scores.append(mse)
    mae_scores.append(mae)

    print(f"Fold {fold+1} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    models.append(model)

    # Predict on all features
    all_pred = model.predict(all_features_flat).flatten()
    all_predictions.append(all_pred)

# Average predictions across folds
y_pred_all = np.mean(all_predictions, axis=0)

# Summary
print("\n=== Cross-Validation Summary ===")
print(f"Mean MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
print(f"Mean MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")


y_pred_all = y_pred_all.squeeze() # (N, 1)-> (N,)

# --- Average Predictions per Time Point (Folder) ---
average_eff_preds = []
std_dev = []
i=0
for folder_name in all_image_folders:
    # Path to embeddings for this folder
    base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5"
    temporary_pred=[]
    for mag in magnifications_to_process:
        images_list_embeddings = listdir(f"{base_embeddings_path}/{mag}")
        for image_file in images_list_embeddings:
            temporary_pred.append(y_pred_all[i])
            i += 1 
    average_eff_preds.append(np.median(temporary_pred)) 
    std_dev.append(np.std(temporary_pred))

average_eff_preds = np.array(average_eff_preds)
std_dev = np.array(std_dev)
all_image_folders=pd.to_datetime(all_image_folders)
df_TSS_eff.index=pd.to_datetime(df_TSS_eff.index)
# --- Calculate Hybrid Model and Uncertainty ---
# Ensure index alignment - crucial if folders were skipped
eff_TSS_hybridmodel = pd.Series(
    #mech_model_output_reindex['.SST_1.X_Out'].values + average_eff_error_preds,
    average_eff_preds,
    index=all_image_folders
)

eff_TSS_hybridmodel_upper = pd.Series(
    eff_TSS_hybridmodel.values + std_dev,
    index=all_image_folders
)
eff_TSS_hybridmodel_lower = pd.Series(
    eff_TSS_hybridmodel.values - std_dev,
    index=all_image_folders
)

# --- Plotting ---
# Use the FOLDER-based split index for plotting

plt.figure(figsize=(10, 4), dpi=150)
plt.plot(df_TSS_eff['LabMeasurement'], '.-', label='Measurements', color='blue')
#plt.plot(mech_model_output_reindex['.SST_1.X_Out'], '-', linewidth=1.5, label='Mechanistic model output', color='green')

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

# ############################################################################ TSS ras

# Flatten inputs
y = y_train_ras # shape (N,)

# Set number of folds
n_splits = 4
kf = KFold(n_splits=n_splits, shuffle=True, random_state=69)

# Store scores
mse_scores = []
mae_scores = []
models=[]
all_predictions=[]
# Start cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_flat)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")
    
    X_tr, X_val = X_flat[train_idx], X_flat[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Define model
    model = Sequential([
        Input(shape=(X_flat.shape[1],)),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_tr, y_tr,
              epochs=10000,
              batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=[early_stop],
              verbose=0)

    # Predict and evaluate
    y_val_pred = model.predict(X_val).squeeze()
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)

    mse_scores.append(mse)
    mae_scores.append(mae)

    print(f"Fold {fold+1} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    models.append(model)

    # Predict on all features
    all_pred = model.predict(all_features_flat).flatten()
    all_predictions.append(all_pred)

# Average predictions across folds
y_pred_all = np.mean(all_predictions, axis=0)
y_pred_ras_all = y_pred_all.squeeze()


# --- Average Predictions per Time Point (Folder) ---
path_to_all_images = 'data/settling_data/filtered_data/images_pileaute_settling_filtered' 
all_image_folders = sorted(listdir(path_to_all_images))
average_ras_preds = []
std_dev_ras = []
i=0
for folder_name in all_image_folders:
    # Path to embeddings for this folder
    base_embeddings_path = f"{path_to_embeddings}/{folder_name}/basin5"
    temporary_pred=[]
    for mag in magnifications_to_process:
        images_list_embeddings = listdir(f"{base_embeddings_path}/{mag}")
        for image_file in images_list_embeddings:
            temporary_pred.append(y_pred_ras_all[i])
            i += 1 
    average_ras_preds.append(np.median(temporary_pred)) 
    std_dev_ras.append(np.std(temporary_pred))

average_ras_preds = np.array(average_ras_preds)
std_dev_ras = np.array(std_dev_ras)
all_image_folders=pd.to_datetime(all_image_folders)
df_TSS_ras.index=pd.to_datetime(df_TSS_ras.index)

# --- Create Hybrid Model and Uncertainty Series ---
ras_TSS_hybridmodel = pd.Series(
    #mech_model_output_reindex['.SST_1.X_Under'].values + average_ras_error_preds,
    average_ras_preds,
    index=all_image_folders # Use the time index from reindexed data
)

# Create the upper and lower bounds for the uncertainty band
ras_TSS_hybridmodel_upper = pd.Series(
    ras_TSS_hybridmodel.values + std_dev_ras,
    index=all_image_folders # Use same time index
)
ras_TSS_hybridmodel_lower = pd.Series(
    ras_TSS_hybridmodel.values - std_dev_ras,
    index=all_image_folders # Use same time index
)

# --- Plotting ---
plt.figure(figsize=(10,4), dpi=150)
plt.plot(df_TSS_ras['LabMeasurement'], '.-', label='Measurements', color='blue')
plt.plot(df_TSS_ras['WeeklyMechModelOutput'], '-', linewidth=1.5, label='Mechanistic model output', color='green') # Commented out as per user code

# Plot hybrid model predictions - use iloc for positional slicing based on plot_split_index
plt.plot(ras_TSS_hybridmodel.iloc[train_indices_folders], '.-', label='HM predictions (train)', color='orange')
plt.plot(ras_TSS_hybridmodel.iloc[test_indices_folders], '.-', label='HM predictions (test)', color='red')

# Plot Standard Deviation Band (Train) - use iloc
plt.fill_between(ras_TSS_hybridmodel.index[train_indices_folders],
                 ras_TSS_hybridmodel_lower.iloc[train_indices_folders],
                 ras_TSS_hybridmodel_upper.iloc[train_indices_folders],
                 color='orange', alpha=0.2, zorder=1)

# Plot Standard Deviation Band (Test) - use iloc
plt.fill_between(ras_TSS_hybridmodel.index[test_indices_folders],
                 ras_TSS_hybridmodel_lower.iloc[test_indices_folders],
                 ras_TSS_hybridmodel_upper.iloc[test_indices_folders],
                 color='red', alpha=0.2, zorder=1)

plt.xlabel("Time")
plt.ylabel("TSS RAS (mg/L)")
plt.legend()
plt.show()
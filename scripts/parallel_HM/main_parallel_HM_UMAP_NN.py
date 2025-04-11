### libraries
from utils.helpers import interpolate_time
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from xgboost import XGBRegressor
from utils.helpers import extract_images

### read TSS effluent lab measurement and mech model outputs. interpolate and calculate error.
start_date = pd.to_datetime("2023-10-01")

lab_measurements = pd.read_csv("data/lab_measurements.txt", sep='\s+', index_col=0)
lab_measurements = lab_measurements.iloc[1:].astype(float)
lab_measurements.index = pd.to_numeric(lab_measurements.index, errors='coerce')
lab_measurements.index = start_date + pd.to_timedelta(lab_measurements.index, unit='D')
lab_measurements = lab_measurements.interpolate(method='time')

mech_model_output = pd.read_csv("data/Project1.Dynamic.Simul.out.txt", sep='\s+', index_col=0)
mech_model_output = mech_model_output.iloc[1:].astype(float)
mech_model_output.index = pd.to_numeric(mech_model_output.index, errors='coerce')
mech_model_output.index = start_date + pd.to_timedelta(mech_model_output.index, unit='D')

path_to_all_images='data/microscope_images' #interpolate to dates on which images were taken
lab_measurements_reindex=interpolate_time(lab_measurements, sorted(listdir(path_to_all_images)))
mech_model_output_reindex=interpolate_time(mech_model_output, sorted(listdir(path_to_all_images)))

TSS_eff_error=lab_measurements_reindex['TSSeffluent'].values-mech_model_output_reindex['.SST_1.X_Out'].values
TSS_ras_error=lab_measurements_reindex['TSSras'].values-mech_model_output_reindex['.SST_1.X_Under'].values

### save labels
all_images = extract_images(path_to_all_images, image_type='all', magnification=10)
all_image_folders = sorted(listdir(path_to_all_images))
i=0
TSSeffluent_labels=[]
TSSras_labels=[]
for folder in all_image_folders:
    path = f"{path_to_all_images}/{folder}/basin5/10x"
    images_list = listdir(path)
    for image in images_list:
        TSSeffluent_labels.append(TSS_eff_error[i]) #give every image a label (=error TSS)
        TSSras_labels.append(TSS_ras_error[i])
    i=i+1

### load the saved embeddings:
X_train=pd.read_csv("data/embeddings/UMAP/umap_embeddings_5D_train.csv", header=None)
all_features=pd.read_csv("data/embeddings/UMAP/umap_embeddings_5D_all.csv", header=None)

#### train model for TSS effluent 
split_index =  len(X_train) 
y_train=TSSeffluent_labels[:split_index]

####################################################################
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# Standaardiseren van de features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
all_features = scaler.transform(all_features)

# Model definiëren (1 enkele dense laag)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=X_train.shape[1])  # 1 output neuron, input_dim = aantal features
])

# Model compileren
model.compile(optimizer='adam', loss='mse')

# Model trainen
import numpy as np
y_train = np.array(y_train)
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# # Model evalueren
# test_loss = model.evaluate(X_test, y_test)
# print(f'Test Loss: {test_loss:.4f}')

# Voorspellingen maken
y_pred = model.predict(all_features)

image_folders = sorted(listdir(path_to_all_images)) 
average_error_preds = []
i=0
for folder in image_folders:
    path = f"{path_to_all_images}/{folder}/basin5/10x"
    images_list = listdir(path)
    temporary=[]
    for image in images_list:
        temporary.append(y_pred[i])
        i+=1
    average_error_preds.append(sum(temporary)/len(temporary))

eff_TSS_hybridmodel=mech_model_output_reindex['.SST_1.X_Out']+np.array(average_error_preds)
eff_TSS_hybridmodel.index = lab_measurements_reindex.index

split_index = int(len(TSS_eff_error) * 0.56)
plt.figure(figsize=(10,4), dpi=150)
plt.plot(lab_measurements_reindex['TSSeffluent'], '.-', label='Measurements', color='blue')
plt.plot(mech_model_output_reindex['.SST_1.X_Out'], '-', linewidth=1.5, label='Mechanistic model output', color='green')
plt.plot(eff_TSS_hybridmodel[:split_index], '.-', label='HM predictions (train)', color='orange')
plt.plot(eff_TSS_hybridmodel[split_index:], '.-', label='HM predictions (test)', color='red')
plt.xlabel("Time")
plt.ylabel("TSS effluent (mg/L)")
plt.legend()
plt.show()


#### train model for TSS RAS 
split_index =  len(X_train) 
y_train2=TSSras_labels[:split_index]

bst2 = XGBRegressor(n_estimators=20000, max_depth=4, learning_rate=0.01, reg_alpha=0.1, reg_lambda=1, objective='reg:squarederror')
bst2.fit(X_train, y_train2)

y_pred2 = bst2.predict(all_features)

image_folders = sorted(listdir(path_to_all_images)) 
average_error_preds = []
i=0
for folder in image_folders:
    path = f"{path_to_all_images}/{folder}/basin5/10x"
    images_list = listdir(path)
    temporary=[]
    print(folder)
    for image in images_list:
        temporary.append(y_pred2[i])
        i+=1
    average_error_preds.append(sum(temporary)/len(temporary))

ras_TSS_hybridmodel=mech_model_output_reindex['.SST_1.X_Under']+average_error_preds
#ras_TSS_hybridmodel.index = lab_measurements_reindex.index

split_index = int(len(TSS_eff_error) * 0.56)
plt.figure(figsize=(10,4), dpi=150)
plt.plot(lab_measurements_reindex['TSSras'], '.-', label='Measurements', color='blue')
plt.plot(mech_model_output_reindex['.SST_1.X_Under'], '-', linewidth=1.5, label='Mechanistic model output', color='green')
plt.plot(ras_TSS_hybridmodel[:split_index], '.-', label='HM predictions (train)', color='orange')
plt.plot(ras_TSS_hybridmodel[split_index:], '.-', label='HM predictions (test)', color='red')
plt.xlabel("Time")
plt.ylabel("TSS RAS (mg/L)")
plt.legend()
plt.show()
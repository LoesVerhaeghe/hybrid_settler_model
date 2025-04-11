import pandas as pd
from os import listdir
from PIL import Image as PImage
import numpy as np

def extract_images(path_to_folders, image_type='all', magnification=10):
    """
    Extract images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        image_type (str): The type of images to extract: 'all', 'old', or 'new', train or test. Default is 'all'.
        old refers to old microscope in the lab, new refers to new microscope
        magnification: type of magnification (10 or 40)

    Returns:
        all_images (list): A list of all extracted images.
    """
    target_folder_microscope_type='2024-01-26'
    target_folder_dataset_type='2024-06-26'
    image_folders = sorted(listdir(path_to_folders)) 

    # Initialize lists for images and labels
    all_images = []

    # Define the condition for folder selection based on image_type
    if image_type == 'all':
        selected_folders = image_folders
    elif image_type == 'old':
        selected_folders = [folder for folder in image_folders if folder < target_folder_microscope_type]
    elif image_type == 'new':
        selected_folders = [folder for folder in image_folders if folder >= target_folder_microscope_type]
    elif image_type == 'train':
        selected_folders = [folder for folder in image_folders if folder <= target_folder_dataset_type]
    elif image_type == 'test':
        selected_folders = [folder for folder in image_folders if folder > target_folder_dataset_type]
    else:
        raise ValueError("Invalid image_type. Choose from 'all', 'old', or 'new' or 'train' or 'test'.")

    selected_folders = sorted(selected_folders)

    # Save all images and labels from the selected folders
    for folder in selected_folders:
        path_to_image = f"{path_to_folders}/{folder}/basin5/{magnification}x"
        images_list = sorted(listdir(path_to_image))
        for image in images_list:
            img = PImage.open(f"{path_to_image}/{image}")  # open in RGB color space
            all_images.append(img)

    return all_images

def extract_images_and_labels(path_to_images, path_to_SVI, image_type='all'):
    """
    Extract images and labels from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        image_type (str): The type of images to extract: 'all', 'old', or 'new'. Default is 'all'.

    Returns:
        all_images (list): A list of all extracted images.
        image_labels (numpy array): A numpy array of the corresponding labels.
    """
    target_folder_microscope_type='2024-01-26'
    target_folder_dataset_type='2024-06-05'
    image_folders = sorted(listdir(path_to_images)) 

    # Initialize lists for images and labels
    all_images = []
    image_labels = []

    # Define the condition for folder selection based on image_type
    if image_type == 'all':
        selected_folders = image_folders
    elif image_type == 'old':
        selected_folders = [folder for folder in image_folders if folder < target_folder_microscope_type]
    elif image_type == 'new':
        selected_folders = [folder for folder in image_folders if folder >= target_folder_microscope_type]
    elif image_type == 'train':
        selected_folders = [folder for folder in image_folders if folder <= target_folder_dataset_type]
    elif image_type == 'test':
        selected_folders = [folder for folder in image_folders if folder > target_folder_dataset_type]
    else:
        raise ValueError("Invalid image_type. Choose from 'all', 'old', or 'new' or 'train' or 'test'.")

    selected_folders = sorted(selected_folders)
      
    # save all SVIs
    all_sheetnames=pd.ExcelFile(path_to_SVI).sheet_names

    SVI=[]
    for sheet_name in all_sheetnames:
        df=pd.read_excel(path_to_SVI, sheet_name=sheet_name, skiprows=[0,1,2,3,4,5,6])
        SVI.append(df['(mL/g)'][0])

    SVI=pd.DataFrame(SVI, columns=['SVI'])
    SVI.index=pd.to_datetime(all_sheetnames)

    # Save all images and labels from the selected folders
    for folder in selected_folders:
        path = f"{path_to_images}/{folder}/basin5/10x"
        images_list = sorted(listdir(path))
        for image in images_list:
            img = PImage.open(f"{path}/{image}")  # open in RGB color space
            all_images.append(img)
            image_labels.append(SVI.loc[folder])

    # Convert image labels to numpy array
    image_labels = np.array(image_labels)

    return all_images, image_labels


def interpolate_time(df, new_index):

    """Return a new DataFrame with all columns values interpolated to the new_index values."""
    # Convert df.index to datetime and then to numerical values (timestamps)
    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    df_index_timestamp = df.index.astype(int) / 10**9  # Convert to seconds since epoch
    
    # Convert new_index to datetime if it contains date strings
    new_index_datetime = pd.to_datetime(new_index)
    new_index_timestamp = new_index_datetime.astype(int) / 10**9  # Convert new_index to timestamps

    # Create an empty DataFrame for output
    df_out = pd.DataFrame(index=new_index_datetime)
    df_out.index.name = df.index.name

    # Interpolate each column
    for colname, col in df.items():
        df_out[colname] = np.interp(new_index_timestamp, df_index_timestamp, col)

    # Convert the index of the interpolated DataFrame back to datetime
    df_out.index = new_index_datetime

    return df_out


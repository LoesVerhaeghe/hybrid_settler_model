import pandas as pd
from os import listdir
from PIL import Image as PImage
import numpy as np

def extract_SBH_columns(file_name, sheet_name):
    """
    Extracts the datacolumns from the excelsheet given the file name and the sheet number. 

    """
    #extract sheet
    df=pd.read_excel(file_name, sheet_name=sheet_name, skiprows=[0,1,2,3,4,5,6])

    #extract time and datacolumns
    df_extracted=pd.DataFrame()
    for i, column_name in enumerate(df.columns):
        if 'SBH' in column_name:
            if df[column_name][1:15].isnull().all():
                pass
            else:
                new_column_name=round(df[column_name][15],3)
                df_extracted[new_column_name]=df[column_name][0:15]
    df_extracted.index=df['Time (min)'][0:15]
    
    #sort the columns from highest concentration to lowest
    sorted_columns = df_extracted.columns.sort_values(ascending=False) 
    df_extracted = df_extracted[sorted_columns]

    return df_extracted

def loadImages(path):
    """ return array of images"""

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + "/" + image)
        loadedImages.append(img)

    return loadedImages


def extract_images(path_to_folders, image_type='all', magnification=10):
    """
    Extract images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        image_type (str): The type of images to extract: 'all', 'old', or 'new'. Default is 'all'.
        magnification: type of magnification (10 or 40)

    Returns:
        all_images (list): A list of all extracted images.
    """
    target_folder='2024-01-26'
    image_folders = listdir(path_to_folders) 

    # Initialize lists for images and labels
    all_images = []

    # Define the condition for folder selection based on image_type
    if image_type == 'all':
        selected_folders = image_folders
    elif image_type == 'old':
        selected_folders = [folder for folder in image_folders if folder < target_folder]
    elif image_type == 'new':
        selected_folders = [folder for folder in image_folders if folder > target_folder]
    else:
        raise ValueError("Invalid image_type. Choose from 'all', 'old', or 'new'.")

    # Save all images and labels from the selected folders
    for folder in selected_folders:
        path_to_image = f"{path_to_folders}/{folder}/basin5/{magnification}x"
        images_list = listdir(path_to_image)
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
    target_folder='2024-01-26'
    image_folders = listdir(path_to_images)

    # Initialize lists for images and labels
    all_images = []
    image_labels = []

    # Define the condition for folder selection based on image_type
    if image_type == 'all':
        selected_folders = image_folders
    elif image_type == 'old':
        selected_folders = [folder for folder in image_folders if folder < target_folder]
    elif image_type == 'new':
        selected_folders = [folder for folder in image_folders if folder > target_folder]
    else:
        raise ValueError("Invalid image_type. Choose from 'all', 'old', or 'new'.")
    
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
        images_list = listdir(path)
        for image in images_list:
            img = PImage.open(f"{path}/{image}")  # open in RGB color space
            all_images.append(img)
            image_labels.append(SVI.loc[folder])

    # Convert image labels to numpy array
    image_labels = np.array(image_labels)

    return all_images, image_labels



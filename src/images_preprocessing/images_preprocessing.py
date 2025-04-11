
from skimage import feature
import skimage as ski
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random
from skimage import img_as_float
from skimage import exposure
import matplotlib.pyplot as plt

def preprocess_images(all_images, size, method=None, flatten=False, show_example=False):
    '''
    Input is dataframe with multiple images. Following preprocessing steps are applied:
    - resized 
    - greyscaled
    - flattened
    - if method is defined: either edges are filtered out, images are normalized or contrast is stretched
    -flatten=True: preprocessed images is flattened
    '''
    preprocessed_imgs = []
    examples = []

    for img in all_images:
        original_img = img.copy()
        img = img.resize(size).convert('L')  # resize and greyscale

        if method == 'edges':
            preprocessed_img = feature.canny(np.array(img), sigma=0.2)

        elif method == 'normalized':
            transform = transforms.Compose([transforms.ToTensor()])
            img_tr = transform(img)
            mean, std = img_tr.mean(), img_tr.std()
            transform_norm = transforms.Compose([transforms.Normalize(mean, std)])   
            img_norm = transform_norm(img_tr)
            preprocessed_img = np.array(img_norm.squeeze().numpy())  # Convert normalized tensor to numpy array

        elif method == 'contrast_stretch':
            img = img_as_float(img)
            p2, p98 = np.percentile(img, (2, 98))
            preprocessed_img = exposure.rescale_intensity(img, in_range=(p2, p98))

        else:
            preprocessed_img = np.array(img)

        preprocessed_imgs.append(preprocessed_img)
        examples.append((original_img, preprocessed_img))  # Save both original and preprocessed images

    image_df = np.array(preprocessed_imgs)
    
    if flatten==True:
        # Flatten image data for DataFrame
        image_data_flat = image_df.reshape(image_df.shape[0], -1)
        # Create column names for the DataFrame
        column_names = [f"Pixel_{i+1}" for i in range(image_data_flat.shape[1])]
        # Create a DataFrame from the image data
        image_df = pd.DataFrame(image_data_flat, columns=column_names)

    # Show examples if requested
    if show_example:
        num_examples = min(5, len(all_images))  # Limit to 5 examples or the number of images if fewer
        example_indices = random.sample(range(len(all_images)), num_examples)

        # Ensure example_indices does not exceed length of examples
        example_indices = [idx for idx in example_indices if idx < len(examples)]

        fig, axs = plt.subplots(len(example_indices), 2, figsize=(8, len(example_indices) * 4))
        for i, idx in enumerate(example_indices):
            original_img, preprocessed_img = examples[idx]
            axs[i, 0].imshow(original_img, cmap='gray')
            #axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(preprocessed_img, cmap='gray')
            #axs[i, 1].set_title('Preprocessed Image')
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

    return image_df

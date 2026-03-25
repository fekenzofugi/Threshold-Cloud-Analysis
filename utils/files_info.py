import os
import re
from datetime import datetime


def get_files_info(directory):
    """
    Get information about files in a specified directory.
    This function ensures the specified directory exists, retrieves a list of all files
    in the directory, and returns the total number of files along with their names.
    Args:
        directory (str): The path to the directory to inspect.
    Returns:
        tuple: A tuple containing:
            - num_files (int): The total number of files in the directory.
            - img_ids (list of str): A list of file names in the directory.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Get the list of all files and directories
    file_list = os.listdir(directory)

    # Filter only files
    file_list = [file for file in file_list if os.path.isfile(os.path.join(directory, file))]

    # Get the number of files
    num_files = len(file_list)

    print(f'Total number of files: {num_files}')

    img_ids = file_list  # Directly assign the list of file names
    
    return num_files, img_ids

# Custom sort key function
def alphanum_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def format_date(product_id, start, end):
    date_string = product_id[start:end]
    date_object = datetime.strptime(date_string, "%Y%m%d").date()
    return date_object


import numpy as np
def add_padding(img, pixels):
    # Get the current size of the image
    current_height, current_width = img.shape[1:3]
    # Define the desired size
    target_height, target_width = current_height + pixels, current_width + pixels

    # Calculate padding
    pad_height = target_height - current_height
    pad_width = target_width - current_width

    # Apply padding
    # Padding on each side: (before, after) for height and width
    # Since the image has a batch dimension (1), we pad along the second and third dimensions
    padded_image = np.pad(img, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    return padded_image
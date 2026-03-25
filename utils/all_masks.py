import os
import re
from datetime import datetime
import requests
import shutil
from retry import retry
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import rasterio
import cv2
import numpy as np
import s2cloudless

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


@retry(tries=10, delay=1, backoff=2)
def getResult(region, image, params, id):
    if params["format"] in ["png", "jpg"]:
        url = image.getThumbURL(
            {
                "region": region,
                "dimensions": params["dimensions"],
                "format": params["format"],
            }
        )
    else:
        url = image.getDownloadURL(
            {
                "region": region,
                "dimensions": params["dimensions"],
                "format": params["format"],
                "bands": params["bands"],
            }
        )

    if params["format"] == "GEO_TIFF":
        ext = "tif"
    else:
        ext = params["format"]

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    out_dir = os.path.abspath(params["out_dir"])
    filename = f"{out_dir}/{id}.{ext}"
    with open(filename, "wb") as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Download Completed: ", id)


COLORS_CLOUDSEN12 = np.array(
    [[0, 0, 0], # clear
    [220, 220, 220], # Thick cloud
    [180, 180, 180], # Thin cloud
    [60, 60, 60]], # cloud shadow
    dtype=np.float32
) / 255
INTERPRETATION_CLOUDSEN12 = ["clear", "Thick cloud", "Thin cloud", "Cloud shadow"]


def plot_segmentation_mask(mask, color_array, interpretation_array=None,legend:bool=True, ax=None):
    cmap_categorical = colors.ListedColormap(color_array)

    norm_categorical = colors.Normalize(vmin=-.5,
                                        vmax=color_array.shape[0]-.5)

    color_array = np.array(color_array)
    if interpretation_array is not None:
        assert len(interpretation_array) == color_array.shape[0], f"Different numbers of colors and interpretation {len(interpretation_array)} {color_array.shape[0]}"


    if ax is None:
        ax = plt.gca()

    ax.imshow(mask, cmap=cmap_categorical, norm=norm_categorical,interpolation='nearest')
    if legend:
        patches = []
        for c, interp in zip(color_array, interpretation_array):
            patches.append(mpatches.Patch(color=c, label=interp))

        ax.legend(handles=patches, fontsize="5", loc='upper right')
    return ax


def plot_cloudmask(mask, legend:bool=True, ax=None):
    return plot_segmentation_mask(mask=mask,color_array=COLORS_CLOUDSEN12,interpretation_array=INTERPRETATION_CLOUDSEN12,legend=legend,ax=ax)


def plot_all_masks(
    out_path, 
    s2, 
    manually, 
    fmask,
    kappa,
    sen2cor,
    s2cloudless_np, 
    scoreplus, 
    senselv, 
    unetmob
):

    # Display all the bands
    fig, ax = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
    rgb = np.moveaxis(s2[[3, 2, 1]], 0, -1) 

    # Plot RGB
    ax[0,0].imshow(rgb)
    ax[0,0].set_title("S2 RGB")
    ax[0,0].axis('off')

    # Plot Manually labeled data
    print(manually.shape)
    plot_cloudmask(manually, ax=ax[0, 1], legend=False)
    ax[0, 1].set_title("Human Labeled")
    ax[0, 1].axis('off')

    # Load Fmask results
    clear = (fmask == 0) | (fmask == 1) | (fmask == 3)
    thick_cloud = fmask == 4
    thin_cloud = fmask*0
    cloud_shadow = fmask == 2

    # apply argmax
    fmask_cloudmask = np.concatenate(
        [clear, thick_cloud, thick_cloud, cloud_shadow],
        axis=0
    ).argmax(axis=0)
    print(fmask_cloudmask.shape)
    plot_cloudmask(fmask_cloudmask, ax=ax[0, 2], legend=False)
    ax[0, 2].set_title("Fmask")
    ax[0, 2].axis('off')

    # Load kappamask_L1C results
    clear = kappa == 1
    thick_cloud = kappa == 4
    thin_cloud = kappa == 3
    cloud_shadow = kappa == 2

    # apply argmax
    kappamask_cloudmask = np.concatenate(
        [clear, thick_cloud, thin_cloud, cloud_shadow],
        axis=0
    ).argmax(axis=0)
    print(kappamask_cloudmask.shape)
    plot_cloudmask(kappamask_cloudmask, ax=ax[1, 0], legend=False)
    ax[1, 0].set_title("KappaMask")
    ax[1, 0].axis('off')

    # Load Sen2Cor results
    # from 11 classes to 4 classes
    thick_cloud = (sen2cor  == 9) | (sen2cor  == 8)
    thin_cloud = (sen2cor  == 10)
    cloud_shadow = (sen2cor  == 3)
    clear = (
        (sen2cor  == 1) | (sen2cor  == 2) | (sen2cor == 4) |
        (sen2cor  == 5) | (sen2cor == 6) | (sen2cor == 7) |
        (sen2cor  == 11)
    )


    # apply argmax
    sen2cor_cloudmask = np.concatenate(
        [clear, thick_cloud, thin_cloud, cloud_shadow],
        axis=0
    ).argmax(axis=0)
    print(sen2cor_cloudmask.shape)
    plot_cloudmask(sen2cor_cloudmask, ax=ax[1, 1], legend=False)
    ax[1, 1].set_title("Sen2Cor")
    ax[1, 1].axis('off')

    # Load s2cloudless results
    s2cloudless_cprob = s2cloudless.S2PixelCloudDetector()
    s2cloudlessnp = s2cloudless_np/100
    s2cloudless_cloudmask = s2cloudless_cprob.get_mask_from_prob(s2cloudlessnp).squeeze()
    print(s2cloudless_cloudmask.shape)
    plot_cloudmask(s2cloudless_cloudmask, ax=ax[1, 2], legend=False)
    ax[1, 2].set_title("s2cloudless")
    ax[1, 2].axis('off')

    # Divide scoreplus into classes
    clear = (scoreplus == 0)
    thick_cloud = (scoreplus == 1)
    thin_cloud = (scoreplus == 2)
    cloud_shadow = (scoreplus == 3)

    # Apply argmax to get the final mask
    scoreplus_cloudmask = np.concatenate(
        [clear, thick_cloud, thin_cloud, cloud_shadow],
        axis=0
    ).argmax(axis=0)

    print(f"{scoreplus_cloudmask.shape} scoreplus")
    plot_cloudmask(scoreplus_cloudmask, ax=ax[2, 0], legend=False)
    ax[2, 0].set_title("score+")
    ax[2, 0].axis('off')


    print(f"{senselv.shape} senselv")
    plot_cloudmask(senselv, ax=ax[2, 1], legend=True)
    ax[2, 1].set_title("senselv")
    ax[2, 1].axis('off')
    

    print(f"{unetmob.shape} unetmob")
    plot_cloudmask(unetmob, ax=ax[2, 2], legend=False)
    ax[2, 2].set_title("unetmob")
    ax[2, 2].axis('off')


    plt.savefig(f"{out_path}/all_masks.png", dpi=300)




def absolute_difference(img1, img2):

    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    #--- take the absolute difference of the images ---
    res = cv2.absdiff(img1, img2)

    #--- convert the result to integer type ---
    res = res.astype(np.uint8)

    #--- find percentage difference based on the number of pixels that are not zero ---
    percentage = 100 - ((np.count_nonzero(res) * 100)/ res.size)

    return round(percentage, 2)


def semantic_difference(img1, img2):

    img1_unique = np.unique(img1)
    img2_unique = np.unique(img2)

    accuracy_dict = {}
    # Calculate accuracy for each pixel value (0, 1, 2, 3)
    for value in range(4):
        # Find pixels with the current value in image1
        mask = (img1 == value)

        # Calculate the number of matching pixels with the same value in both images
        matching_pixels = np.sum(mask & (img2 == value))

        # Total number of pixels with the current value in image1
        total_pixels = np.sum(mask)

        # Calculate accuracy as a percentage
        accuracy = (matching_pixels / total_pixels * 100) if total_pixels > 0 else 0

        # Store the accuracy in the dictionary
        accuracy_dict[value] = round(accuracy, 2)

    return accuracy_dict


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

def calculate_pixel_accuracy(manually, model_np, model_name, model_acc):
    pixel_accuracy = semantic_difference(manually, model_np)
    return {
        "Model": model_name,
        "Clear": pixel_accuracy[0],
        "Thick Cloud": pixel_accuracy[1],
        "Thin Cloud": pixel_accuracy[2],
        "Cloud Shadow": pixel_accuracy[3],
        "Absolute": model_acc
    }

def get_np(path, crs):
    src = rasterio.open(
        path,
        mode = 'r',
        driver = 'GTiff',
        crs = crs,
    )
    return src.read()
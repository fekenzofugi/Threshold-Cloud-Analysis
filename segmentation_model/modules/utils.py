import pandas as pd
import numpy as np
import h5py
import ast
import os
import datetime
import segmentation_model.modules.classify_hard as ch
    
#GID
def int_to_ascii(n):
    n = int(n)
    if 0 <= n <= 9:
        return chr(n + 48)
    elif 10 <= n <= 35:
        return chr(n + 55)
    elif 36 <= n <= 61:
        return chr(n + 61)
    else:
        raise ValueError(f"Must be in the range [0, 61] , but is {n}")

def ascii_to_int(c):
    c = ord(c)
    if 48 <= c <= 57:
        return c - 48
    elif 65 <= c <= 90:
        return c - 55
    elif 97 <= c <= 122:
        return c - 61
    else:
        raise ValueError("Must be in the range [0-9, A-Z, a-z]")
    
def int_to_b62(n):
    if n == 0:
        return "00"
    s = ""
    while n > 0:
        s = int_to_ascii(n % 62) + s
        n = n // 62
    return s.zfill(2)

def b62_to_int(s):
    n = 0
    for i, c in enumerate(s):
        n += ascii_to_int(c) * (62 ** (len(s) - i - 1))
    return n

def generate_GID(img_name, BASE_DATE : datetime.date) -> str:
    """
    This function returns the GID of the object.
    """
    name = img_name.split('_')
    #roi = name[1]
    #date = name[5][:-4]
    #tile = name[3]
    roi = name[1]
    date = name[5][:8]
    tile = name[3]
    gid = int_to_ascii(int(roi)) + int_to_ascii(int(tile)) + int_to_ascii((datetime.datetime.strptime(date, '%Y%m%d').date() - BASE_DATE).days // 5)
    return gid

# export to csv

def export_to_csv(img, img_name : str, objects : list[dict], filename : str, BASE_DATE : datetime.date):
    """
    This function exports the classification of the objects to a csv file.
    Format:
    GID, class
    """
    create_csv(filename)
    data = []
    existing_df = pd.read_csv(filename, index_col=0)
    i = 0
    for obj in objects: #Generate a new funticion to classify and GID the objects
        GID = generate_GID(img_name, BASE_DATE) + int_to_b62(i)
        CLASS = ch.hard_classify(img, obj)
        # Save the GID in the object
        obj['GID'] = GID
        # Save the class in the object
        obj['class'] = CLASS
        VALIDATED = 0
        data.append([GID, CLASS, VALIDATED])
        i += 1
    df = pd.DataFrame(data, columns=['GID', 'class', 'validated'])
    df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(filename, index=True)

def create_csv(filename : str):
    """
    This function creates a csv file with the header 'GID, class'.
    """
    if not os.path.exists(filename):
        data = []
        open(filename, 'w').close()
        data.append(['00000', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])    
        df = pd.DataFrame(data, columns=['GID', 'PMLI', 'NDVI', 'BSI', 'BSPI', 'NDWI', 'NDMI', 'APGI', 'CLOUD', 'SHADOW', 'class', 'validated'])
        df.to_csv(filename, index=True)

# save masks

def save_masks_and_info_as_hdf5(masks, output_path, img_name):
    with h5py.File(os.path.join(output_path, f"{img_name}_masks_info.h5"), 'w') as hf:
        print(f"Saving masks and info for {img_name} in {output_path}")
        for i, mask in enumerate(masks):
            # Convert mask['segmentation'] to a NumPy array with a specific data type
            segmentation_array = np.array(mask['segmentation'])#, dtype=np.uint8)
            hf.create_dataset(f"mask_{i}_segmentation", data=segmentation_array)
            hf.create_dataset(f"mask_{i}_area", data=np.array(mask['area'], dtype=np.float32))
            hf.create_dataset(f"mask_{i}_class", data=np.array(mask['class'], dtype=np.float32))
            hf.create_dataset(f"mask_{i}_GID", data=np.string_(mask['GID']))
            hf.create_dataset(f"mask_{i}_bbox", data=np.array(mask['bbox'], dtype=np.float32))

def load_masks_and_info_from_hdf5(output_path, img_name):
    masks = []
    with h5py.File(os.path.join(output_path, f"{img_name}_masks_info.h5"), 'r') as hf:
        i = 0
        while f"mask_{i}_segmentation" in hf:
            mask = {
                'segmentation': hf[f"mask_{i}_segmentation"][:],
                'area': hf[f"mask_{i}_area"][()],
                'class': hf[f"mask_{i}_class"][()],
                'GID': hf[f"mask_{i}_GID"].asstr()[()],
                'bbox': hf[f"mask_{i}_bbox"][:],
            }
            masks.append(mask)
            i += 1
    return masks

# base date

def get_base_date(img_name : str) -> datetime.date:
    """
    This function returns the base date of the images.
    """
    name = img_name.split('_')
    date = name[5][:-4] 
    return datetime.datetime.strptime(date, '%Y%m%d').date()

# get img list
def get_img_list(folder):
    names = []
    for filename in os.listdir(folder):
        if filename.endswith(".tif"):
            names.append(filename)
    return names

import segmentation_models_pytorch as smp
import requests
import torch
import rasterio
import numpy as np
import os
from utils.files_info import get_files_info, alphanum_key

def apply_unetmob(path):
    # Download the model if not already present
    model_filename = "UNetMobV2_V2.pt"
    model_url = f"https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus/resolve/main/demo/models/{model_filename}"

    if not os.path.exists(model_filename):
        with requests.get(model_url, stream=True) as r:
            with open(model_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, classes=4, in_channels=13)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval().to(device)

    # Get file info
    num_files, ids = get_files_info(path)
    ids = sorted(ids, key=alphanum_key)

    for i in range(num_files):
        file_path = os.path.join(path, ids[i])

        with rasterio.open(file_path) as src:
            scene = src.read()
            meta = src.meta.copy()

            # Normalize and prepare input
            sentinel2 = scene[:12] / 10000.0
            dummy_band = np.zeros((1, sentinel2.shape[1], sentinel2.shape[2]), dtype=sentinel2.dtype)
            input_tensor = np.concatenate([sentinel2, dummy_band], axis=0)

            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(device)

            with torch.no_grad():
                prediction = model(input_tensor).argmax(dim=1).squeeze().cpu().numpy().astype(scene.dtype)

            # Add the predicted mask as a new band
            scene_with_mask = np.concatenate([scene, prediction[np.newaxis, :, :]], axis=0)

        # Update metadata
        meta.update({
            'count': scene_with_mask.shape[0],
            'dtype': scene_with_mask.dtype
        })

        # Overwrite the file with new data
        with rasterio.open(file_path, 'w', **meta) as dst:
            for b in range(scene_with_mask.shape[0]):
                dst.write(scene_with_mask[b], b + 1)

        print(f"Overwritten with mask: {file_path}")

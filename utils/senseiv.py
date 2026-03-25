import rasterio
from senseiv2.inference import CloudMask
from senseiv2.utils import get_model_files
from senseiv2.constants import SENTINEL2_DESCRIPTORS

#    All the credits for this code go to the authors of the SEnSeIv2 model.
#    https://github.com/aliFrancis/SEnSeIv2
#    https://ieeexplore.ieee.org/document/10505181


def apply_senseiv(path, out_path, crs, device="cpu"):

    src = rasterio.open(
        path,
        mode = 'r',
        driver = 'GTiff',
        crs = crs,
    )

    scene = src.read()
    print(scene.shape)

    input = scene / 10000

    # Pick pre-trained model from https://huggingface.co/aliFrancis/SEnSeIv2
    # model_name = 'SEnSeIv2-SegFormerB2-alldata-ambiguous'
    model_name = 'SegFormerB2-S2-unambiguous'
    # model_name = 'SEnSeIv2-DeepLabv3-S2-unambiguous'
    # model_name = 'SEnSeIv2-SegFormerB2-S2-ambiguous'
    config, weights = get_model_files(model_name)

    # Lots of options in the kwargs for different settings
    model = CloudMask(config, weights, verbose=True, categorise=True, device=device)

    senselv = model(input, descriptors=SENTINEL2_DESCRIPTORS, stride=357).squeeze()

    scene = senselv.cpu().numpy()

    bands = scene.shape[0]

    # Open a new dataset for writing
    new_dataset = rasterio.open(
        f"{out_path}/senseiv.tif",
        mode='w',
        driver='GTiff',
        height=input.shape[1],
        width=input.shape[2],
        count=bands,
        dtype=rasterio.uint8,  # Change data type to uint8
        crs=src.crs,  # Ensure this is appropriate for your data
        transform=src.transform,
        compress='lzw'  # Add compression to reduce file size
    )

    try:
        for j in range(bands):
            new_dataset.write(scene.astype(rasterio.uint8), j + 1)  # Ensure data is cast to uint8
    finally:
        new_dataset.close()
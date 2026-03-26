# Assessing the Impact of Threshold on Cloud Masking Algorithms for Downstream Tasks

This repository contains the code for the ML4RS 4th Edition paper titled "Assessing the Impact of Threshold on Cloud Masking Algorithms for Downstream Tasks". The paper investigates the influence of cloud masking threshold on the performance of downstream tasks, in particular full-scene segmentation with SAM.

## Repository Structure
`ML4RS_RESULTS.ipynb` - contains the source code to reproduce the results presented in the paper. It includes data loading and evaluation steps.

`segmentation_model` - contains the code of SAM-based full-scene segmentation model used in the paper.

`utils` - contains utility functions for data processing and evaluation.

`threshold_data` - contains the results already generated for different thresholds, which can be used to reproduce the results without needing to run the full process.

`s2_ids.txt` - contains the list of Sentinel-2 image IDs used in the study.

`requirements.txt` - lists the required Python packages to run the code.

## Data
The data used in this study is from `CloudSEN12+` dataset, which is a collection of Sentinel-2 images with cloud masks. The dataset can be accessed [here](https://huggingface.co/datasets/tacofoundation/cloudsen12/tree/main). You can either load the dataset remotely using `tacoreader` or download the metadata files and load them locally for faster access.

```python
# Remotely load the Cloud-Optimized Dataset 
dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")
dataset_extra = tacoreader.load("tacofoundation:cloudsen12-extra")
```

To download the files locally, you can visit the [CloudSEN12+ dataset page](https://huggingface.co/datasets/tacofoundation/cloudsen12/tree/main) and download the metadata files for both `cloudsen12-l1c` and `cloudsen12-extra`. After downloading, you can load the dataset locally as follows:

```python
# https://huggingface.co/datasets/tacofoundation/cloudsen12/tree/main and download CLOUDSEN12+ metadata files.
# WAY FASTER THAN REMOTE LOADING.
dataset = tacoreader.load([
    "dataset/cloudsen12-l1c.0000.part.taco", 
    "dataset/cloudsen12-l1c.0001.part.taco", 
    "dataset/cloudsen12-l1c.0002.part.taco",
    "dataset/cloudsen12-l1c.0003.part.taco", 
    "dataset/cloudsen12-l1c.0004.part.taco",])

dataset_extra = tacoreader.load([
    "dataset/cloudsen12-extra.0000.part.taco", 
    "dataset/cloudsen12-extra.0001.part.taco", 
    "dataset/cloudsen12-extra.0002.part.taco",])
```

## SAM model

Click the link below to download the checkpoint for the SAM model used in the paper:
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)

Then add the model to the `/Checkpoints` directory.

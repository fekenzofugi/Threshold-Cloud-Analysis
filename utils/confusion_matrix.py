import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import rasterio
import matplotlib.pyplot as plt

def compute_confusion_matrix(outpath, true_labels, predicted_labels, class_names, model_title="Model"):
    """
    Compute and plot the confusion matrix for segmentation classes.

    Parameters:
    - true_labels: np.array, ground truth labels
    - predicted_labels: np.array, predicted labels
    - class_names: list of str, names of the classes
    - model_title: str, title of the model to be displayed on the plot

    Returns:
    - cm: np.array, confusion matrix
    """
    print(range(len(class_names)))
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
    plt.figure(figsize=(20, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', labelpad=20, fontsize=12)
    plt.ylabel('True', labelpad=20, fontsize=12)
    plt.title(f'Confusion Matrix - {model_title}', fontsize=18)
    plt.savefig(f'{outpath}/{model_title}_confusion_matrix.png')
    return cm

def process_directory(directory_path, model='unetmob'):
    predicted_labels = []
    true_labels = []
    
    directory_names = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    print("Directory Names:", directory_names)

    for dir_name in directory_names:
        try:
            with rasterio.open(f"{directory_path}/{dir_name}/{model}.tif", mode='r', driver='GTiff', crs='EPSG:4326') as src:
                scene = src.read()
                print(np.unique(scene))
                predicted_labels.append(scene)
                print(scene.shape)

            with rasterio.open(f"{directory_path}/{dir_name}/manually.tif", mode='r', driver='GTiff', crs='EPSG:4326') as src:
                scene = src.read()[0]
                true_labels.append(scene)
                print(np.unique(scene))
                print(scene.shape)

        except rasterio.errors.RasterioIOError as e:
            print(f"Failed to open {directory_path}/{dir_name}/{model}.tif: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    return true_labels, predicted_labels

if __name__ == "__main__":
    directory_path = 'data'
    model = 'senseiv'
    true_labels, predicted_labels = process_directory(directory_path, model=model)

    # Flatten the lists of arrays
    true_labels = np.concatenate(true_labels).flatten()
    predicted_labels = np.concatenate(predicted_labels).flatten()

    class_names = ['Land', 'Thick Cloud', 'Thin Cloud', 'Shadow']

    cm = compute_confusion_matrix(true_labels, predicted_labels, class_names, model_title=model)
    print("Confusion Matrix:\n", cm)

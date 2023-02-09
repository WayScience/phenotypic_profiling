# 6. Single Cell Images

In this module, we use the model on single cell images to clearly demonstrate its application.

### Single Cell Sample Image Dataset

The [single cell sample image data](mitocheck_single_cell_sample_images) have kindly been provided by Dr. Thomas Walter of the MitoCheck consortium.
This dataset contains sample single cell images in the following format:
```
mitocheck_single_cell_sample_images
│
└───phenotypic_class
│   │
│   └───sample_image_path.png

```

Because the features for these cells have already been extracted in [`mitocheck_data`](https://github.com/WayScience/mitocheck_data), we do not re-extract features from these images in this module.
Instead, features are associated with a single cell image based on the cell's location metadata (plate, well, frame, x, y).

### Top 5 Performing Classes

In [correct_15_images.ipynb](correct_15_images.ipynb), we show 15 sample single cell images that the final model from [2.train_model](../2.train_model/) correctly classifies.
Three single cell images from each of the 5 top performing classes (as determined by F1 score from [compiled_F1_scores.tsv](../3.evaluate_model/evaluations/F1_scores/compiled_F1_scores.tsv)) are displayed and their paths are saved in [top_5_performing_classes.tsv](../6.single_cell_images/sample_image_paths/top_5_performing_classes.tsv).

## Step 1: Extract Sample Image Data

Use the commands below to run the Jupyter notebooks and extract the sample image data:

```sh
# Make sure you are located in 6.single_cell_images
cd 6.single_cell_images

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Interpret model
bash single_cell_images.sh
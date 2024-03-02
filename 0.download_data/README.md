# Download Data

In this module, we download labeled datasets from `mitocheck_data`.

### Download/Preprocess Data

Complete instructions for data download and preprocessing are located at: https://github.com/WayScience/mitocheck_data

### Usage

In this repository, we download all labeled data from a version controlled [mitocheck_data](https://github.com/WayScience/mitocheck_data).
We specify the path to each set of `mitocheck_data` with `labeled_data_paths` in [download_data.ipynb](download_data.ipynb).

### Data Preview

The labeled dataset includes CellProfiler (CP) and DeepProfiler (DP) features as well as metadata (location, perturbation, etc) for cells from the original MitoCheck project.
The breakdown of cell counts by manually-assigned phenotypic class for the `ic` (illumination corrected) dataset is as follows:

| Phenotypic Class    | Cell Count |
|---------------------|-------|
| Interphase          | 420   |
| Polylobed           | 367   |
| Prometaphase        | 345   |
| OutOfFocus          | 304   |
| Apoptosis           | 273   |
| Binuclear           | 184   |
| MetaphaseAlignment  | 175   |
| SmallIrregular      | 164   |
| Hole                | 114   |
| Elongated           | 110   |
| ADCCM               | 95    |
| Anaphase            | 84    |
| Large               | 79    |
| Grape               | 74    |
| Metaphase           | 74    |
| Folded              | 54    |

**Note**: The `get_features_data()` function (defined in [split_utils.py](../utils/split_utils.py)) used to load the labeled cell dataset excludes cells from the `Folded` phenotypic class when loading the labeled cells.
In our testing, the low representation of `Folded` cells leads to significantly low classification accuracy for this class (only tested with multi-class models).
Thus, we opt to exclude these cells from all training and testing.

## Step 1: Download Data

Use the commands below to download labeled training dataset:

```sh
# Make sure you are located in 0.download_data
cd 0.download_data

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Download data
bash download_data.sh
```

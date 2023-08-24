# Download Data

In this module, a labeled dataset is downloaded from `mitocheck_data`.

### Download/Preprocess Data

Complete instructions for data download and preprocessing can be found at: https://github.com/WayScience/mitocheck_data

### Usage

In this repository, all labeled data is downloaded from a version controlled [mitocheck_data](https://github.com/WayScience/mitocheck_data).

The version of mitocheck_data used is specified by the hash corresponding to a current commit.
The current hash being used is `e1f86cd007657f8247310b78df92891b22e51621` which corresponds to [mitocheck_data/e1f86cd](https://github.com/WayScience/mitocheck_data/tree/e1f86cd007657f8247310b78df92891b22e51621).
The `hash` variable can be set in [download_data.ipynb](download_data.ipynb) to change which version of mitocheck_data is being accessed.

### Data Preview

The labeled dataset includes CellProfiler (CP) and DeepProfiler (DP) features as well as metadata (location, perturbation, etc) for cells from the original MitoCheck project.
The breakdown of cell counts by phenotypic class (as labeled manually by MitoCheck) is as follows:

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
In our testing, the low representation of `Folded` cells leads to significantly low classification accuracy for this class.
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

"""Utils for the single-cell image module (finding sample images, displaying images, etc)"""


import pathlib
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.linear_model import LogisticRegression


def get_sample_image_metadata(frame_details: str) -> dict:
    """
    get frame metadata from features samples movie details string

    Parameters
    ----------
    frame_details : str
        string from name of sample image file
        ex: PLLT0010_27--ex2005_05_13--sp2005_03_23--tt17--c5___P00173_01___T00082___X0397___Y0618____img.png

    Returns
    -------
    dict
        dictionary with plate, well_num, frame, x, y location data
    """

    # parse location data from sample image file name
    plate = frame_details.split("--")[0].replace("PL", "")
    well_num = int(frame_details.split("___")[1][1:6])
    frame = int(frame_details.split("___")[2][1:6]) + 1
    x = int(frame_details.split("___")[3][1:6])
    y = int(frame_details.split("___")[4][1:6])

    # return location metadata as dictionary
    return {
        "Metadata_Plate": plate,
        "Metadata_Well": well_num,
        "Metadata_Frame": frame,
        "Location_Center_X": x,
        "Location_Center_Y": y,
    }


def get_sample_image_path(
    cell_phenotypic_class: str,
    cell_location_metadata: dict,
    single_cell_images_dir_path: pathlib.Path,
) -> pathlib.Path:
    """
    Given a cell's phenotypic class and its location metadata, try to find the sample image path from a directory of sample images

    Parameters
    ----------
    cell_phenotypic_class : str
        phenotypic class of cell we are trying to find sample image of
    cell_location_metadata : dict
        location metadata of cell we are trying to find sample image of
    single_cell_images_dir_path : pathlib.Path
        path to sample images directory

    Returns
    -------
    pathlib.Path
        path to sample image or None if sample image cannot be found
    """
    # iterate through the first structure in sample images directory (folder of each phenotypic_class)
    for phenotypic_class_dir_path in single_cell_images_dir_path.iterdir():
        # iterate through second structure in sample images directory (sample images of each cell)
        for sample_image_path in phenotypic_class_dir_path.iterdir():
            phenotypic_class = phenotypic_class_dir_path.name
            # get sample image location metadata for the particular sample image
            sample_image_metadata = get_sample_image_metadata(sample_image_path.name)

            # if the cell of interest matches location data with sample image
            # error margin for x and y is 20 (IDR_stream and MitoCheck center locations are slightly different)
            if (
                cell_phenotypic_class == phenotypic_class
                and cell_location_metadata["Metadata_Plate"]
                == sample_image_metadata["Metadata_Plate"]
                and cell_location_metadata["Metadata_Well"]
                == sample_image_metadata["Metadata_Well"]
                and cell_location_metadata["Metadata_Well"]
                == sample_image_metadata["Metadata_Well"]
                and abs(
                    cell_location_metadata["Location_Center_X"]
                    - sample_image_metadata["Location_Center_X"]
                )
                < 20
                and abs(
                    cell_location_metadata["Location_Center_Y"]
                    - sample_image_metadata["Location_Center_Y"]
                )
                < 20
            ):
                return sample_image_path

    # if no matching sample image is found, return None
    return None


def get_class_sample_images(
    phenotypic_class: str,
    dataset: pd.DataFrame,
    log_reg_model: LogisticRegression,
    single_cell_images_dir_path: pathlib.Path,
    num_images: int = 3,
    correct: bool = True,
) -> list:
    """
    get a list of sample image paths for a phenotypic class corresponding to whether or not the model correctly predicted the image

    Parameters
    ----------
    phenotypic_class : str
        true class of cell to find
    dataset : pd.DataFrame
        cell dataset with metadata and features
    log_reg_model : LogisticRegression
        model used to classify cells
    single_cell_images_dir_path : pathlib.Path
        path to single-cell sample images
    num_images : int, optional
        number of image paths to return, by default 3
    correct : bool, optional
        whether or not the model has correctly predicted the image, by default True

    Returns
    -------
    list
        list of image paths
    """

    # list of image paths to be returned
    sample_image_paths = []

    # subset of data that only contains cells of desired phenotypic class
    class_data = dataset.loc[
        (dataset["Mitocheck_Phenotypic_Class"] == phenotypic_class)
    ]
    # columns of dataset that contain feature data
    feature_columns = [column for column in dataset.columns if "efficientnet" in column]

    for _, cell in class_data.iterrows():
        # get model phenotypic class prediction for this cell
        cell_features = cell[feature_columns].to_numpy().reshape(1, -1)
        cell_class_prediction = log_reg_model.predict(cell_features)[0]

        # try to find an image from sample images that matches this cell
        cell_location_metadata = {
            "Metadata_Plate": cell["Metadata_Plate"],
            "Metadata_Well": cell["Metadata_Well"],
            "Metadata_Frame": cell["Metadata_Frame"],
            "Location_Center_X": cell["Location_Center_X"],
            "Location_Center_Y": cell["Location_Center_Y"],
        }
        sample_image_path = get_sample_image_path(
            phenotypic_class, cell_location_metadata, single_cell_images_dir_path
        )

        # if a sample image path was able to be found for this cell
        if sample_image_path is not None:
            # only append the image path if the cell prediction matched the desired prediction result (correct/incorrect)
            if (cell_class_prediction == phenotypic_class) == correct:
                sample_image_paths.append(sample_image_path)

        # if the correct number of image paths have been found
        if len(sample_image_paths) == num_images:
            return sample_image_paths

    return sample_image_paths


def get_15_correct_sample_images(
    phenotypic_classes: list,
    dataset: pd.DataFrame,
    log_reg_model: LogisticRegression,
    single_cell_images_dir_path: pathlib.Path,
) -> pd.DataFrame:
    """
    get 15 accurately predicted sample images, 3 for each 5 phenotypic classes given as inputs

    Parameters
    ----------
    phenotypic_classes : list
        list of phenotypic classes to get 3 correct sample images for
    dataset : pd.DataFrame
        cell dataset with metadata and features
    log_reg_model : LogisticRegression
        model used to classify cells
    single_cell_images_dir_path : pathlib.Path
        path to single-cell sample images

    Returns
    -------
    pd.DataFrame
        dataframe with phenotypic classes and paths to 3 sample images that the model correctly predicted
    """
    compiled_sample_images = []

    for phenotypic_class in phenotypic_classes:
        # get images of desired phenotypic class
        sample_image_paths = get_class_sample_images(
            phenotypic_class,
            dataset,
            log_reg_model,
            single_cell_images_dir_path,
            correct=True,
        )
        # add these paths to compiled sample images
        sample_image_paths_dataframe = pd.Series(sample_image_paths).to_frame().T
        compiled_sample_images.append(sample_image_paths_dataframe)

    # convert compiled sample images to dataframe and add prefix to column names
    compiled_sample_images = pd.concat(compiled_sample_images).add_prefix("Path_")
    # add phenotypic classes to compiled sample images dataframe
    compiled_sample_images.insert(
        loc=0, column="Phenotypic_Class", value=phenotypic_classes
    )

    return compiled_sample_images.reset_index(drop=True)


def plot_15_correct_sample_images(sample_images_df: pd.DataFrame):
    """
    show the 15 correct sample images collected with get_15_correct_sample_images()

    Parameters
    ----------
    sample_images_df : pd.DataFrame
        dataframe with phenotypic classes and paths to 3 sample images that the model accurately predicted
    """
    path_columns = [column for column in sample_images_df.columns if "Path_" in column]

    # create 3x5 plot for plotting images
    fig, axs = plt.subplots(3, 5)
    fig.set_size_inches(14, 8)
    ax_x = 0
    ax_y = 0

    # iterate through sample images data to plot each class sample images
    for _, sample_images_data in sample_images_df.iterrows():
        paths = sample_images_data[path_columns].to_list()

        for path in paths:
            sample_image = mpimg.imread(path)

            axs[ax_x, ax_y].imshow(sample_image, cmap="gray")

            # only add pehnotypic class to images at the top of the figure
            if ax_x == 0:
                axs[ax_x, ax_y].set_title(sample_images_data["Phenotypic_Class"])

            # adjust the "coordinates" of the figure subplot we are editing
            # if we reach the end of the subplot row (each row has 3 subplots), move down to next row (reset x coord, add 1 to y coord)
            ax_x += 1
            if ax_x == 3:
                ax_x = 0
                ax_y += 1

    # remove axis numbers/ticks
    for ax in axs.flat:
        ax.axis("off")

    # add title to entire figure
    fig.suptitle("Correctly Predicted Sample Image for Top 5 Performing Classes")

    plt.show()

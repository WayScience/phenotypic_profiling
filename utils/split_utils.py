import pandas as pd
import numpy as np
import pathlib

# set numpy seed to make random operations reproduceable
np.random.seed(0)

def get_features_data(load_path: pathlib.Path) -> pd.DataFrame:
    """get features data from csv at load path
    Args:
        load_path (pathlib.Path): path to training data csv
    Returns:
        pd.DataFrame: training dataframe
    """
    # read dataset into pandas dataframe
    features_data = pd.read_csv(load_path, index_col=0)

    # remove fold class that has low representation
    features_data = features_data[
        features_data["Mitocheck_Phenotypic_Class"] != "Folded"
    ]

    return features_data


def get_image_indexes(training_data: pd.DataFrame, images: list) -> list:

    image_indexes_list = []
    for image in images:
        image_indexes = training_data.index[
            training_data["Metadata_Plate_Map_Name"] == image
        ].tolist()
        image_indexes_list.extend(image_indexes)

    return image_indexes_list


def get_random_images_indexes(training_data: pd.DataFrame, num_images: int) -> list:
    """get ramdom images from training dataset
    Args:
        training_data (pd.DataFrame): pandas dataframe of training data
        num_images (int): number of images to holdout
    Returns:
        List: list of unique images for holding out
    """
    unique_images = pd.unique(training_data["Metadata_Plate_Map_Name"])
    images = np.random.choice(unique_images, size=num_images, replace=False)

    return images


def get_intelligent_images(training_data: pd.DataFrame, num_images: int) -> list:
    """get images from training dataset and try to balance labels present in these images
    add an image if it has at least class not represented by the other images
    if the image doesn't have a contribution via new class, try a new image
    Args:
        training_data (pd.DataFrame): pandas dataframe of training data
        num_images (int): number of images to holdout
    Returns:
        List: list of unique images with intelligently balanced phenotypic classes
    """
    remaining_images = pd.unique(training_data["Metadata_Plate_Map_Name"])
    remaining_classes = pd.unique(training_data["Mitocheck_Phenotypic_Class"]).tolist()

    images = []

    for image_number in range(num_images):
        image_contributes = False
        image = ""
        while not image_contributes:
            image = np.random.choice(remaining_images, size=1, replace=False)[0]
            image_data = training_data.loc[
                (training_data["Metadata_Plate_Map_Name"] == image)
            ]

            if len(remaining_classes) == 0:
                image_contributes = True

            image_classes = pd.unique(image_data["Mitocheck_Phenotypic_Class"])
            # check if any phenotypic class from the current image is in the remaining classes
            if any(
                phenotypic_class in image_classes
                for phenotypic_class in remaining_classes
            ):
                image_contributes = True
                remaining_classes = [
                    x for x in remaining_classes if x not in image_classes
                ]

        images.append(image)
        remaining_images = np.delete(
            remaining_images, np.where(remaining_images == image)
        )

    return images


def get_representative_images(
    training_data: pd.DataFrame, num_images: int, attempts: int = 100
) -> list:
    """get images from training dataset and such that every phenotypic class is represented
    returns None if no combintation of images are found that represent every phenotypic class within number of trials
    Args:
        training_data (pd.DataFrame): pandas dataframe of training data
        num_images (int): number of images to holdout
        attempts (int): number of times to try getting representative images
    Returns:
        List: list of images with every phenotypic class represented or None if this list cannot be curated
    """
    unique_classes = pd.unique(training_data["Mitocheck_Phenotypic_Class"]).tolist()

    trial = 0
    while trial < attempts:
        images = get_intelligent_images(training_data, num_images)

        images_data = pd.DataFrame()
        for image in images:
            image_data = training_data.loc[
                (training_data["Metadata_Plate_Map_Name"] == image)
            ]
            images_data = pd.concat([images_data, image_data])

        unique_image_classes = pd.unique(
            images_data["Mitocheck_Phenotypic_Class"]
        ).tolist()
        if set(unique_image_classes) == set(unique_classes):
            return images

        trial += 1

    print("No combination of images found that represents all classes!")
    return None
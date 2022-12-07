"""
utilities for combining MitoCheck datasets from 2006 and 2015
"""

import pandas as pd
import numpy as np


def combine_datasets(
    training_data_2006: pd.DataFrame, training_data_2015: pd.DataFrame
) -> pd.DataFrame:
    """
    combine 2006 and 2015 datasets without cell repeats

    Parameters
    ----------
    training_data_2006 : pd.DataFrame
        training dataset from 2006
    training_data_2015 : pd.DataFrame
        training dataset from 2015, includes object outline data

    Returns
    -------
    pd.DataFrame
        dataframe with single cell data from 2006 and 2015 without duplicates
    """
    non_duplicate_2015 = []

    for index_2015, row_2015 in training_data_2015.iterrows():
        in_both = False
        for _, row_2006 in training_data_2006.iterrows():
            # check if cell from 2015 data is in same plate, well, frame as 2006 data and in the same location (within cell outline)
            if (
                row_2015["Metadata_Plate"] == row_2006["Metadata_Plate"]
                and row_2015["Metadata_Well"] == row_2006["Metadata_Well"]
                and row_2015["Metadata_Frame"] == row_2006["Metadata_Frame"]
            ):
                # if x and y coordinates from cells are within 5 pixels of each other, that same cell is in both datasets
                cell_location_2006 = [
                    int(row_2006["Location_Center_X"]),
                    int(row_2006["Location_Center_Y"]),
                ]
                cell_location_2015 = [
                    int(row_2015["Location_Center_X"]),
                    int(row_2015["Location_Center_Y"]),
                ]
                if (
                    abs(cell_location_2006[0] - cell_location_2015[0]) < 5
                    and abs(cell_location_2006[1] - cell_location_2015[1]) < 5
                ):
                    print(
                        f"Cell found in both datasets! Index {index_2015} in 2015 data."
                    )
                    in_both = True

        if not in_both:
            non_duplicate_2015.append(row_2015.drop(labels=["Object_Outline"]))

    non_duplicate_2015 = pd.DataFrame(non_duplicate_2015)
    return pd.concat([training_data_2006, non_duplicate_2015], ignore_index=True)

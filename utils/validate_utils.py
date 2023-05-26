"""
utilities for validating logistic regression models on MitoCheck single-cell dataset
"""

import pandas as pd


def create_classification_profiles(
    plate_classifications_dir_link: str, cell_line_plates: dict
) -> pd.DataFrame:
    """
    create classsification profiles for correlation to cell health label profiles
    a classification profile consists of classification probabilities averaged across perturbation and cell line

    Parameters
    ----------
    plate_classifications_dir_link : str
        raw GitHub link to plate classifications directory
    cell_line_plates : dict
        cell line names, each with a list of plate names that are correlated to their respective cell line

    Returns
    -------
    pd.DataFrame
        classification profiles dataframe
    """

    cell_line_classification_profiles = []

    for cell_line in cell_line_plates:
        # create one large dataframe for all 3 plates in the particular cell line
        cell_line_plate_names = cell_line_plates[cell_line]
        cell_line_plate_classifications = []
        for cell_line_plate_name in cell_line_plate_names:
            plate_classifications_link = f"{plate_classifications_dir_link}/{cell_line_plate_name}_cell_classifications.csv.gz"
            plate_classifications = pd.read_csv(
                plate_classifications_link, compression="gzip", index_col=0
            )
            cell_line_plate_classifications.append(plate_classifications)
        cell_line_plate_classifications = pd.concat(
            cell_line_plate_classifications, axis=0
        ).reset_index(drop=True)

        # create dataframe with cell classifications averaged across perturbation, include cell line metadata
        # add cell line metadata
        cell_line_plate_classifications["Metadata_cell_line"] = cell_line
        # rename perturbation column to match the format of cell health label profiles, in this case "perturbation" corresponds to "reagent" because DeepProfiler (used much earlier in pipeline) makes no distinction
        cell_line_plate_classifications = cell_line_plate_classifications.rename(
            columns={"Metadata_Reagent": "Metadata_pert_name"}
        )

        # get rid of extra metadata columns
        phenotypic_classes = [
            col
            for col in cell_line_plate_classifications.columns.tolist()
            if ("Metadata" not in col)
            and (col not in ["Location_Center_X", "Location_Center_Y"])
        ]
        columns_to_keep = [
            "Metadata_pert_name",
            "Metadata_cell_line",
        ] + phenotypic_classes
        cell_line_plate_classifications = cell_line_plate_classifications[
            columns_to_keep
        ]

        # average across perturbation
        cell_line_classification_profile = cell_line_plate_classifications.groupby(
            ["Metadata_pert_name", "Metadata_cell_line"]
        ).mean()
        cell_line_classification_profiles.append(cell_line_classification_profile)

    classification_profiles = pd.concat(cell_line_classification_profiles, axis=0)
    # convert pandas index to columns
    classification_profiles = classification_profiles.reset_index(
        level=["Metadata_pert_name", "Metadata_cell_line"]
    )
    # convert no reagent to empty in pert column to follow format of cell health label profiles
    classification_profiles = classification_profiles.replace(
        {"Metadata_pert_name": "no reagent"}, "EMPTY"
    )

    return classification_profiles

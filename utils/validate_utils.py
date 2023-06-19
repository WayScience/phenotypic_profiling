"""
utilities for validating logistic regression models on MitoCheck single-cell dataset
"""

import pathlib

import pandas as pd
import numpy as np
from ccc.coef import ccc
from scipy.spatial.distance import squareform


def create_classification_profiles(
    plate_classifications_dir: pathlib.Path, cell_line_plates: dict
) -> pd.DataFrame:
    """
    create classsification profiles for correlation to cell health label profiles
    a classification profile consists of classification probabilities averaged across perturbation and cell line

    Parameters
    ----------
    plate_classifications_dir : pathlib.Path
        path to plate classifications directory
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
            plate_classifications = pathlib.Path(
                f"{plate_classifications_dir}/{cell_line_plate_name}__cell_classifications.csv.gz"
            )
            plate_classifications = pd.read_csv(
                plate_classifications, compression="gzip", index_col=0
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


def get_cell_health_corrs(
    final_profile_dataframe: pd.DataFrame, corr_type: str, cell_line: str
):
    # remove metadata columns
    cleaned_final_profile_dataframe = final_profile_dataframe.drop(
        columns=["Metadata_profile_id", "Metadata_pert_name", "Metadata_cell_line"]
    )

    if corr_type == "pearson":
        corrs = cleaned_final_profile_dataframe.corr(method="pearson")
    if corr_type == "ccc":
        corrs = ccc(cleaned_final_profile_dataframe)
        corrs = squareform(corrs)
        np.fill_diagonal(corrs, 1.0)
        corrs = pd.DataFrame(
            corrs,
            index=cleaned_final_profile_dataframe.columns.tolist(),
            columns=cleaned_final_profile_dataframe.columns.tolist(),
        )

    # refactor correlations to easily readable format (from 70 cell health indicators)
    corrs = corrs.iloc[70:, :70]
    corrs.index.name = "phenotypic_class"
    corrs = corrs.reset_index()

    # add metadata to correlations
    corrs["cell_line"] = cell_line
    corrs["corr_type"] = corr_type

    return corrs


def get_tidy_long_corrs(final_profile_dataframe: pd.DataFrame):
    compiled_tidy_long_corrs = []

    # get correlations for all cell lines
    all_pearson_corrs = get_cell_health_corrs(final_profile_dataframe, "pearson", "all")
    all_ccc_corrs = get_cell_health_corrs(final_profile_dataframe, "ccc", "all")
    compiled_tidy_long_corrs += [all_pearson_corrs, all_ccc_corrs]

    for cell_line in final_profile_dataframe["Metadata_cell_line"].unique():
        # get subset of final profile dataframe for the particular cell line
        cell_line_profiles = final_profile_dataframe.loc[
            final_profile_dataframe["Metadata_cell_line"] == cell_line
        ]
        # get correlations for this cell line subset
        cell_line_pearson_corrs = get_cell_health_corrs(
            cell_line_profiles, "pearson", cell_line
        )
        cell_line_ccc_corrs = get_cell_health_corrs(cell_line_profiles, "ccc", cell_line)
        compiled_tidy_long_corrs += [cell_line_pearson_corrs, cell_line_ccc_corrs]

    # combine all correlations
    compiled_tidy_long_corrs = pd.concat(compiled_tidy_long_corrs)
    
    # convert wide dataframe to tidy long format
    compiled_tidy_long_corrs = pd.melt(
        compiled_tidy_long_corrs,
        id_vars=["phenotypic_class", "cell_line", "corr_type"],
        var_name="cell_health_indicator",
        value_name="corr_value",
    )

    return compiled_tidy_long_corrs

import pandas as pd
import pathlib


def create_classification_profiles(
    plate_classifications_dir_link: str, cell_line_plates: dict
) -> pd.DataFrame:
    cell_line_classification_profiles = []

    for cell_line in cell_line_plates:
        print(f"Creating classification profiles for cell line {cell_line}")
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

        # create dataframe with cell classifications averaged across pertubation, include cell line metadata
        print("Averaging classification across perturbation metadata...")
        # add cell line metadata, rename pertubation column
        cell_line_plate_classifications["Metadata_cell_line"] = cell_line
        cell_line_plate_classifications = cell_line_plate_classifications.rename(
            columns={"Metadata_Reagent": "Metadata_pert_name"}
        )

        # get rid of extra metadata columns
        phenotypic_classes = [
            col
            for col in cell_line_plate_classifications.columns.tolist()
            if "Metadata" not in col
        ]
        phenotypic_classes.remove("Location_Center_X")
        phenotypic_classes.remove("Location_Center_Y")
        columns_to_keep = [
            "Metadata_pert_name",
            "Metadata_cell_line",
        ] + phenotypic_classes
        cell_line_plate_classifications = cell_line_plate_classifications[
            columns_to_keep
        ]

        # average across pertubation
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

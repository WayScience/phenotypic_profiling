library(ggplot2)
library(dplyr)

focus_phenotypes <- c(
    "Apoptosis",
    "Elongated",
    "Interphase",
    "Large",
    "Metaphase",
    "OutOfFocus"
)

focus_phenotype_colors <- c(
    "Apoptosis" = "#e7298a",
    "Elongated" = "#1b9e77",
    "Interphase" = "#7570b3",
    "Large" = "#d95f02",
    "Metaphase" = "black",
    "OutOfFocus" = "red",
    "Other" = "grey"
)

focus_phenotype_labels <- c(
    "Apoptosis" = "Apoptosis",
    "Elongated" = "Elongated",
    "Interphase" = "Interphase",
    "Large" = "Large",
    "Metaphase" = "Metaphase",
    "OutOfFocus" = "OutOfFocus",
    "Other" = "Other"
)

feature_spaces <- c(
    "CP" = "CellProfiler",
    "DP" = "DeepProfiler",
    "CP_and_DP" = "CP and DP",
    "CP_zernike_only" = "CP Zernike",
    "CP_areashape_only" = "CP AreaShape"
)

facet_labels <- c(
    "CP" = "CellProfiler",
    "DP" = "DeepProfiler",
    "CP_and_DP" = "CP and DP"
)

data_split_colors <- c(
    "test" = "#D64933",
    "train" = "#044389"
)
data_split_labels <- c(
    "test" = "Test",
    "train" = "Train"
)
shuffled_linetypes <- c(
    "False" = "solid",
    "True" = "dashed"
)
shuffled_labels <- c(
    "False" = "False",
    "True" = "True"
)

focus_corr_colors = c(
    "TRUE" = "blue",
    "FALSE" = "orange"
)
focus_corr_labels  = c(
    "TRUE" = "Yes",
    "FALSE" = "No"
)

feature_type_with_data_split_colors <- c(
    "CPtest" = "#1b9e77",
    "DPtest" = "#d95f02",
    "CP_and_DPtest" = "#7570b3",

    "CPtrain" = "#c8e9df",
    "DPtrain" = "#fedbcd",
    "CP_and_DPtrain" = "#cbd4e7"
)

feature_type_with_data_split_labels <- c(
    "CP_and_DPtest" = "CP + DP (Test)",
    "CP_and_DPtrain" = "CP + DP (Train)",
    "CPtest" = "CP (Test)",
    "CPtrain" = "CP (Train)",
    "DPtest" = "DP (Test)",
    "DPtrain" = "DP (Train)"
)

subset_feature_type_with_data_split_colors <- c(
    "CP_areashape_onlytest" = "#1f78b4",
    "CP_zernike_onlytest" = "#7beb1a",

    "CP_areashape_onlytrain" = "#aacee6",
    "CP_zernike_onlytrain" = "#b2df8a"
)

subset_feature_type_with_data_split_labels <- c(
    "CP_areashape_onlytest" = "AreaShape only (Test)",
    "CP_zernike_onlytest" = "Zernike only (Test)",

    "CP_areashape_onlytrain" = "AreaShape only (Train)",
    "CP_zernike_onlytrain" = "Zernike only (Train)"
)

phenotypic_ggplot_theme <- (
    theme_bw()
    + theme(
        axis.text = element_text(size = 10),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10),
        legend.key.height = unit(0.8, "lines"),
        strip.text = element_text(size = 10),
        strip.background = element_rect(
            linewidth = 0.5, color = "black", fill = "#fdfff4"
        )
    )
)

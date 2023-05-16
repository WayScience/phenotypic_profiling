# Set consistent figure themes and colors
suppressPackageStartupMessages(library(ggplot2))

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

figure_theme <- (
    theme_bw()
    + theme(
        axis.text = element_text(size = 7),
        axis.title = element_text(size = 10),
        legend.text = element_text(size = 9),
        legend.title = element_text(size = 10),
        strip.text = element_text(size = 8),
        strip.background = element_rect(
            colour = "black",
            fill = "#fdfff4"
            )
    )
)


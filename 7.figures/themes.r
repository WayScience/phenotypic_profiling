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

phenotype_categories <- list(
    "Interphase" = c("Interphase", "Elongated", "Large"),
    "Mitosis" = c("Prometaphase", "MetaphaseAlignment", "Metaphase", "Anaphase"),
    "Mitotic conseq." = c("Binuclear", "Polylobed", "Grape"),
    "Dynamic changes" = c("Hole", "SmallIrregular", "Folded"),
    "Other" = c("Apoptosis", "OutOfFocus", "ADCCM")
)

facet_labels <- c(
    "CP" = "CellProfiler",
    "DP" = "DeepProfiler",
    "CP_and_DP" = "CP and DP"
)

feature_space_labels <- c(
    "CP" = "CellProfiler",
    "DP" = "DeepProfiler",
    "CP_and_DP" = "CP and DP"
)

feature_space_colors <- c(
    "CP" = "#1b9e77",
    "DP" = "#d95f02",
    "CP_and_DP" = "#7570b3"
)


focus_corr_colors <- c(
    "TRUE" = "blue",
    "FALSE" = "orange"
)

focus_corr_labels <- c(
    "TRUE" = "Yes",
    "FALSE" = "No"
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

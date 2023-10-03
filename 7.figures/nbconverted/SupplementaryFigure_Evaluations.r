suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))

# Load variables important for plotting (e.g., themes, phenotypes, etc.)
source("themes.r")

figure_dir <- "figures"
output_sup_fig_scm <- file.path(figure_dir, "supplementary", "supplementary_singleclassmodel_evaluations.png")
output_sup_fig_loio <- file.path(figure_dir, "supplementary", "supplementary_leaveoneimageout_evaluations.png")

# Load data
results_dir <- file.path(
    "..", "3.evaluate_model", "evaluations", "precision_recall_curves"
)
results_file <- file.path(results_dir, "compiled_SCM_PR_curves.tsv")

pr_df <- readr::read_tsv(
    results_file,
    col_types = readr::cols(
        .default = "d",
        "Model_Class" = "c",
        "data_split" = "c",
        "shuffled" = "c",
        "feature_type" = "c"
    )
) %>%
    dplyr::select(!`...1`) %>%
    dplyr::mutate(feature_type_with_data_split = paste0(feature_type, data_split)) %>%
    dplyr::filter(!stringr::str_detect(Model_Class, "Negative"))
    

print(dim(pr_df))
head(pr_df)

pr_curve_gg <- (
    ggplot(pr_df, aes(x = Recall, y = Precision))
    + geom_line(aes(color = feature_type_with_data_split, linetype = shuffled))
    + facet_wrap("~Model_Class", nrow = 3)
    + theme_bw()
    + xlab("Recall")
    + ylab("Precision")
    + scale_color_manual(
        name = "Single class\nmodel scenario",
        labels = feature_type_with_data_split_labels,
        values = feature_type_with_data_split_colors
    )
    + scale_linetype_manual(
        name = "Shuffled\ntraining\ndata",
        labels = shuffled_labels,
        values = shuffled_linetypes
    )
    + guides(
        color = guide_legend(order = 1),
        linetype = guide_legend(order = 2),
    )
    + phenotypic_ggplot_theme
    # Decrease spacing in legend
    + theme(
        legend.spacing.y = unit(0.1, "cm"),
        legend.box.spacing = unit(0.2, "cm"),
        legend.key.size = unit(0.7, "lines"),
        legend.key.width = unit(1, "lines"),
        axis.text.x = element_text(angle = 90),
        strip.text = element_text(size = 8.5),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 14)
    )
)

pr_curve_gg

# Load data
results_dir <- file.path(
    "..", "3.evaluate_model", "evaluations", "F1_scores"
)
results_file <- file.path(results_dir, "compiled_SCM_F1_scorescoress.tsv")

f1_score_df <- readr::read_tsv(
    results_file,
    col_types = readr::cols(
        .default = "d",
        "Phenotypic_Class" = "c",
        "data_split" = "c",
        "shuffled" = "c",
        "feature_type" = "c"
    )
) %>%
    dplyr::select(!`...1`) %>%
    dplyr::mutate(feature_type_with_data_split = paste0(feature_type, data_split)) %>%
    dplyr::filter(data_split == "test") %>%
    dplyr::filter(!stringr::str_detect(Phenotypic_Class, "Negative")) %>%
    dplyr::filter(Phenotypic_Class != "Weighted")

f1_score_df$Phenotypic_Class <-
    factor(f1_score_df$Phenotypic_Class, levels = rev(unique(f1_score_df$Phenotypic_Class)))

print(dim(f1_score_df))
head(f1_score_df)

# How many of the phenotypic classes are certain feature types the best?
top_scores_df <- f1_score_df %>%
    dplyr::arrange(desc(F1_Score)) %>%
    dplyr::group_by(Phenotypic_Class) %>%
    dplyr::slice(1)

print(table(top_scores_df$feature_type_with_data_split))

top_scores_df

f1_score_gg <- (
    ggplot(f1_score_df, aes(x = Phenotypic_Class, y = F1_Score))
    + geom_bar(aes(fill = feature_type_with_data_split), stat = "identity", position = "dodge")
    + theme_bw()
    + xlab("Single cell phenotype\n(Single class models)")
    + ylab("F1 Score (Test set)")
    + scale_fill_manual(
        name = "Model scenario",
        labels = feature_type_with_data_split_labels,
        values = feature_type_with_data_split_colors
    )
    + phenotypic_ggplot_theme
    # Decrease spacing in legend and rotate textfigure
    + theme(
        legend.spacing.y = unit(0.2, "cm"),
        legend.box.spacing = unit(0.3, "cm"),
        legend.key.size = unit(0.8, "lines"),
        legend.key.width = unit(1, "lines"),
        axis.text.x = element_text(angle = 90),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 14)
    )
    + coord_flip()
)

f1_score_gg

sup_fig <- (
    pr_curve_gg | 
    f1_score_gg
) + plot_layout(widths = c(3, 0.7)) + plot_annotation(tag_levels = "A")

ggsave(output_sup_fig_scm, dpi = 500, height = 6, width = 15)

sup_fig

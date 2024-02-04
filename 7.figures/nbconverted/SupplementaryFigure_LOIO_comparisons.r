suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))

# Load variables important for plotting (e.g., themes, phenotypes, etc.)
source("themes.r")

# Set I/O
results_dir <- file.path("..", "3.evaluate_model", "evaluations", "LOIO_probas")

# Results from `get_LOIO_probabilities.ipynb`
results_file <- file.path(results_dir, "compiled_LOIO_probabilities.tsv")

# Results from `LOIO_evaluation.ipynb`
results_summary_file <- file.path(results_dir, "LOIO_summary_ranks.tsv.gz")
results_summary_perphenotype_file <- file.path(results_dir, "LOIO_summary_ranks_perphenotype.tsv.gz")

output_fig_loio <- file.path("figures", "supplementary", "loio_supplement.png")

# Set custom labellers for adding context to facet text plotting
shuffled_labeller <- function(value) {
  paste("Shuffled:", value)
}

balanced_labeller <- function(value) {
  paste("Model:", value)
}

# Create phenotype categories dataframe
# Note: phenotype_categories defined in themes.r
phenotype_categories_df <- stack(phenotype_categories) %>%
  rename(Mitocheck_Category = ind, Mitocheck_Phenotypic_Class = values)

phenotype_categories_df

loio_df <- readr::read_tsv(
    results_file,
    col_types = readr::cols(
        .default = "d",
        "Model_Phenotypic_Class" = "c",
        "Mitocheck_Phenotypic_Class" = "c",
        "Cell_UUID" = "c",
        "Metadata_DNA" = "c",
        "Model_type" = "c",
        "Balance_type" = "c",
        "Dataset_type" = "c",
        "Model_Feature_Type" = "c"
    )
) %>%
    dplyr::select(!`...1`) %>%
    dplyr::group_by(
        Cell_UUID,
        Model_type,
        Balance_type,
        Metadata_DNA,
        Mitocheck_Phenotypic_Class,
        Model_Feature_Type,
        Dataset_type,
        Model_C,
        Model_l1_ratio
    ) %>%
    dplyr::mutate(rank_value = rank(desc(Predicted_Probability))) %>%
    dplyr::mutate(correct_pred = paste(Mitocheck_Phenotypic_Class == Model_Phenotypic_Class)) %>%
    dplyr::left_join(
        phenotype_categories_df,
        by = "Mitocheck_Phenotypic_Class"
    ) %>%
    dplyr::left_join(
        phenotype_categories_df,
        by = c("Model_Phenotypic_Class" = "Mitocheck_Phenotypic_Class"),
        suffix = c("", "_model")
    ) %>%
    dplyr::mutate(
        correct_class_pred = paste(Mitocheck_Category == Mitocheck_Category_model)
    )

loio_df$rank_value <- factor(loio_df$rank_value, levels = paste(sort(unique(loio_df$rank_value))))

# The `feature_spaces` variable is defined in themes.r
loio_df$Model_Feature_Type <-
    dplyr::recode_factor(loio_df$Model_Feature_Type, !!!feature_spaces)

refactor_logical <- c("TRUE" = "TRUE", "FALSE" = "FALSE")
loio_df$correct_pred <-
    dplyr::recode_factor(loio_df$correct_pred, !!!refactor_logical)

loio_df$Shuffled <- dplyr::recode_factor(
    loio_df$Model_type,
    "final" = "FALSE", "shuffled_baseline" = "TRUE"
)

print(dim(loio_df))
head(loio_df, 5)

# Focus main result on select LOIO parameters:
# Balanced model, with ic, select feature spaces (CP, DP, CP_and_DP)
loio_focus_df <- loio_df %>%
    dplyr::filter(
        Balance_type == "balanced",
        Dataset_type == "ic",
        Model_Feature_Type %in% c("CellProfiler", "DeepProfiler", "CP and DP")
    )

loio_feature_space_gg <- (
    ggplot(loio_focus_df,
        aes(x = rank_value, y = Predicted_Probability)
          )
    + geom_boxplot(aes(fill = correct_pred), outlier.size = 0.1, lwd = 0.3)
    + theme_bw()
    + phenotypic_ggplot_theme
    + facet_grid(
        "Shuffled~Model_Feature_Type",
        labeller = labeller(Shuffled = shuffled_labeller)
    )
    + labs(x = "Rank of prediction", y = "Prediction probability")
    + scale_fill_manual(
        "Correct\nphenotype\nprediction?",
        values = focus_corr_colors,
        labels = focus_corr_labels
    )
)

loio_feature_space_gg

# Load per image, per phenotype, per feature space summary
loio_summary_per_phenotype_df <- readr::read_tsv(
    results_summary_file,
    col_types = readr::cols(
        .default = "d",
        "Metadata_DNA" = "c",
        "Model_type" = "c",
        "Mitocheck_Phenotypic_Class" = "c",
        "Balance_type" = "c",
        "Dataset_type" = "c",
        "Model_Feature_Type" = "c"
    )
) %>%
    dplyr::mutate(loio_label = "Leave one image out") %>%
    # Generate a new column that we will use for plotting
    # Note, we define focus_phenotypes in themes.r
    dplyr::mutate(Mitocheck_Plot_Label = if_else(
        Mitocheck_Phenotypic_Class %in% focus_phenotypes,
        Mitocheck_Phenotypic_Class,
        "Other"
    ))

loio_summary_per_phenotype_df$Model_Feature_Type <-
    dplyr::recode_factor(loio_summary_per_phenotype_df$Model_Feature_Type, !!!feature_spaces)

loio_summary_per_phenotype_df$Mitocheck_Plot_Label <-
    dplyr::recode_factor(loio_summary_per_phenotype_df$Mitocheck_Plot_Label, !!!focus_phenotype_labels)

loio_summary_per_phenotype_df$Shuffled <- dplyr::recode_factor(
    loio_summary_per_phenotype_df$Model_type,
    "final" = "FALSE", "shuffled_baseline" = "TRUE"
)

head(loio_summary_per_phenotype_df, 3)

length(unique(loio_summary_per_phenotype_df$Metadata_DNA))

percent_summary_df <- loio_summary_per_phenotype_df %>%
    dplyr::mutate(pass_on_average = Average_Rank < 2) %>%
    dplyr::filter(Shuffled == FALSE) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_Feature_Type,
        Balance_type,
        Model_type,
        Dataset_type
    ) %>%
    dplyr::mutate(total_pass = sum(pass_on_average), total_count = n()) %>%
    dplyr::mutate(
        percent_pass = round((total_pass / total_count) * 100, 1)
    ) %>%
    dplyr::select(
        Mitocheck_Phenotypic_Class,
        Model_Feature_Type,
        Balance_type,
        Model_type,
        Shuffled,
        Dataset_type,
        percent_pass,
        total_pass,
        total_count
    ) %>%
    dplyr::distinct() %>%
    dplyr::ungroup() %>%
    dplyr::mutate(add_plot_text = paste0(total_pass, "/", total_count, "\n", "(", percent_pass, "%)"))

head(percent_summary_df)

dim(loio_summary_per_phenotype_df %>%
    dplyr::filter(Mitocheck_Phenotypic_Class == "Polylobed", Model_Feature_Type == "CellProfiler", Balance_type == "balanced", Model_type == "final", Dataset_type == "ic"))

percent_summary_all_phenotypes_df <- loio_summary_per_phenotype_df %>%
    dplyr::mutate(pass_on_average = Average_Rank < 2) %>%
    dplyr::group_by(
        Model_Feature_Type,
        Balance_type,
        Model_type,
        Dataset_type
    ) %>%
    dplyr::mutate(total_pass = sum(pass_on_average), total_count = n()) %>%
    dplyr::mutate(
        percent_pass = round((total_pass / total_count) * 100, 1)
    ) %>%
    dplyr::select(
        Model_Feature_Type,
        Balance_type,
        Model_type,
        Dataset_type,
        percent_pass,
        total_pass,
        total_count
    ) %>%
    dplyr::distinct() %>%
    dplyr::ungroup() %>%
    dplyr::mutate(add_plot_text = paste0(total_pass, "/", total_count, "\n", "(", percent_pass, "%)")) %>%
    dplyr::arrange(desc(percent_pass))

head(percent_summary_all_phenotypes_df)

ic_comparison_summary_df <- percent_summary_df %>%
    dplyr::filter(
        Model_Feature_Type %in% c("CellProfiler", "DeepProfiler", "CP and DP")
    ) %>%
    tidyr::pivot_wider(
        names_from = Dataset_type,
        values_from = total_pass,
        id_cols = c(Mitocheck_Phenotypic_Class, Model_Feature_Type, Balance_type, Shuffled)
    ) %>%
    dplyr::mutate(ic_impact = ic - no_ic)

# Reverse order of predicted label for plotting
ic_comparison_summary_df$Mitocheck_Phenotypic_Class <-
    factor(
        ic_comparison_summary_df$Mitocheck_Phenotypic_Class,
        levels = rev(sort(unique(paste(ic_comparison_summary_df$Mitocheck_Phenotypic_Class))))
    )

ic_comparison_summary_df

ic_comparison_results_gg <- (
    ggplot(ic_comparison_summary_df, aes(x = ic_impact, y = Mitocheck_Phenotypic_Class))
    + geom_bar(stat = "identity")
    + facet_grid(
        "Model_Feature_Type~Balance_type",
        labeller = labeller(Shuffled = shuffled_labeller, Balance_type = balanced_labeller)
    )
    + theme_bw()
    + phenotypic_ggplot_theme
    + labs(x = "Difference in number of images correctly predicted\nin LOIO after adding illumination correction ", y = "Mitocheck phenotypes")
    + geom_vline(xintercept = 0, linetype = "dashed", color = "red")
)

ic_comparison_results_gg

compiled_fig <- (
    wrap_elements(loio_feature_space_gg) / wrap_elements(ic_comparison_results_gg)
) + plot_layout(heights = c(0.5, 1)) + plot_annotation(tag_levels = "A")

ggsave(output_fig_loio, dpi = 500, height = 12, width = 10)

compiled_fig

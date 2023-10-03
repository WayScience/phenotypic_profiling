suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))

# Load variables important for plotting (e.g., themes, phenotypes, etc.)
source("themes.r")

# Set I/O
results_dir <- file.path("..", "3.evaluate_model", "evaluations", "LOIO_probas")

results_file <- file.path(results_dir, "compiled_LOIO_probabilites.tsv")
results_summary_file <- file.path(results_dir, "LOIO_summary_ranks.tsv")
results_summary_perphenotype_file <- file.path(results_dir, "LOIO_summary_ranks_perphenotype.tsv")

output_fig_loio <- file.path("figures", "main_figure_4_loio.png")

# Set constants
high_threshold <- 0.9

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
        "Model_Feature_Type" = "c"
    )
) %>%
    dplyr::select(!`...1`) %>%
    dplyr::group_by(
        Cell_UUID,
        Metadata_DNA,
        Mitocheck_Phenotypic_Class,
        Model_Feature_Type,
        Model_C,
        Model_l1_ratio
    ) %>%
    dplyr::mutate(rank_value = rank(desc(Predicted_Probability))) %>%
    dplyr::mutate(correct_pred = paste(Mitocheck_Phenotypic_Class == Model_Phenotypic_Class)) %>%
    dplyr::left_join(phenotype_categories_df, by = "Mitocheck_Phenotypic_Class") %>%
    dplyr::left_join(phenotype_categories_df, by = c("Model_Phenotypic_Class" = "Mitocheck_Phenotypic_Class"), suffix = c("", "_model")) %>%
    dplyr::mutate(correct_class_pred = paste(Mitocheck_Category == Mitocheck_Category_model))

loio_df$rank_value <- factor(loio_df$rank_value, levels = paste(sort(unique(loio_df$rank_value))))

loio_df$Model_Feature_Type <-
    dplyr::recode_factor(loio_df$Model_Feature_Type, !!!facet_labels)

refactor_logical <- c("TRUE" = "TRUE", "FALSE" = "FALSE")
loio_df$correct_pred <-
    dplyr::recode_factor(loio_df$correct_pred, !!!refactor_logical)

print(dim(loio_df))
head(loio_df, 5)

loio_feature_space_gg <- (
    ggplot(loio_df,
        aes(x = rank_value, y = Predicted_Probability)
          )
    + geom_boxplot(aes(fill = correct_pred), outlier.size = 0.1, lwd = 0.3)
    + theme_bw()
    + phenotypic_ggplot_theme
    + facet_grid("~Model_Feature_Type")
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
        "Mitocheck_Phenotypic_Class" = "c",
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

feature_order <- c("CP" = "CP", "DP" = "DP", "CP_and_DP" = "CP_and_DP")

loio_summary_per_phenotype_df$Model_Feature_Type <-
    dplyr::recode_factor(loio_summary_per_phenotype_df$Model_Feature_Type, !!!feature_order)

loio_summary_per_phenotype_df$Mitocheck_Plot_Label <-
    dplyr::recode_factor(loio_summary_per_phenotype_df$Mitocheck_Plot_Label, !!!focus_phenotype_labels)

head(loio_summary_per_phenotype_df, 3)

per_image_category_gg <- (
    ggplot(loio_summary_per_phenotype_df, aes(x = Average_Rank, y = Average_P_Value))
    + geom_point(aes(size = Count, color = Model_Feature_Type), alpha = 0.2)
    + theme_bw()
    + phenotypic_ggplot_theme
    + facet_grid("Mitocheck_Plot_Label")
    + labs(
        x = "Average rank of correct label\n(per held out image)",
        y = "Average probability of correct label\n(per held out image)"
    )
    + scale_color_manual(
        name = "Feature space",
        labels = facet_labels,
        values = facet_colors
    )
    + scale_size_continuous(
        name = "Per image\ncell count"
    )
    + geom_vline(xintercept=2, linetype = "dashed", color = "blue")
    + theme(
        strip.text.y = element_text(size = 8.3),
    )
    + guides(
        color = guide_legend(
            order = 1,
            override.aes = list(size = 2, alpha = 1)
        ),
        linetype = guide_legend(
            order = 2,
            override.aes = list(alpha = 1)
        ),
    )
)

per_image_category_gg

phenotypic_class_category_counts <- loio_df %>%
    dplyr::select(Mitocheck_Phenotypic_Class, correct_pred, correct_class_pred) %>%
    dplyr::group_by(Mitocheck_Phenotypic_Class, correct_pred, correct_class_pred) %>%
    dplyr::summarize(phenotype_count = n()) %>%
    dplyr::ungroup()

loio_thresh_df <- loio_df %>%
    dplyr::mutate(pass_threshold = paste(Predicted_Probability >= high_threshold)) %>%
    dplyr::group_by(Mitocheck_Phenotypic_Class, correct_pred, correct_class_pred, pass_threshold) %>%
    dplyr::summarize(count = n()) %>%
    dplyr::left_join(phenotypic_class_category_counts, by = c("Mitocheck_Phenotypic_Class", "correct_pred", "correct_class_pred")) %>%
    dplyr::mutate(phenotype_prop = count / phenotype_count)

phenotypic_class_counts <- loio_df %>%
    dplyr::select(Mitocheck_Phenotypic_Class, correct_pred) %>%
    dplyr::group_by(Mitocheck_Phenotypic_Class, correct_pred) %>%
    dplyr::summarize(phenotype_count = n()) %>%
    dplyr::ungroup()

loio_thresh_df <- loio_df %>%
    dplyr::mutate(pass_threshold = paste(Predicted_Probability >= high_threshold)) %>%
    dplyr::group_by(Mitocheck_Phenotypic_Class, correct_pred, pass_threshold) %>%
    dplyr::summarize(count = n()) %>%
    dplyr::left_join(phenotypic_class_counts, by = c("Mitocheck_Phenotypic_Class", "correct_pred")) %>%
    dplyr::mutate(phenotype_prop = count / phenotype_count)

# Reverse order of predicted label for plotting
loio_thresh_df$Mitocheck_Phenotypic_Class <-
    factor(loio_thresh_df$Mitocheck_Phenotypic_Class, levels = rev(unique(loio_thresh_df$Mitocheck_Phenotypic_Class)))

head(loio_thresh_df)

custom_labeller <- function(value) {
  paste("Correct prediction:\n", value)
}

correct_pred_proportion_gg <- (
    ggplot(
        loio_thresh_df,
        aes(
            x = phenotype_prop,
            y = Mitocheck_Phenotypic_Class,
            fill = pass_threshold
        )
    )
    + geom_bar(stat = "identity")
    + geom_text(
        data = loio_thresh_df %>%
            dplyr::filter(pass_threshold == TRUE),
        color = "black",
        aes(label = count),
        nudge_x = 0.07,
        size = 3
    )
    + facet_wrap("~correct_pred", labeller = labeller(correct_pred = custom_labeller))
    + theme_bw()
    + phenotypic_ggplot_theme
    + theme(axis.text = element_text(size = 7.5))
    + scale_fill_manual(
        paste0("Does cell\npass strict\nthreshold?\n(p = ", high_threshold, ")"),
        values = focus_corr_colors,
        labels = focus_corr_labels,
        breaks = c("TRUE", "FALSE")
    )
    + labs(x = "Cell proportions", y = "Mitocheck phenotypes")
)

correct_pred_proportion_gg

same_class_wrong_pred_summary_df <- loio_df %>%
    dplyr::filter(correct_pred == "FALSE") %>%
    dplyr::filter(rank_value == 1) %>%
    dplyr::group_by(Mitocheck_Phenotypic_Class, correct_class_pred) %>%
    dplyr::summarize(count = n(), avg_prob = mean(Predicted_Probability)) %>%
    dplyr::ungroup() %>%
    dplyr::left_join(phenotype_categories_df, by = "Mitocheck_Phenotypic_Class") %>%
    dplyr::group_by(Mitocheck_Category, correct_class_pred) %>%
    dplyr::summarize(total_count = sum(count), avg_prob = mean(avg_prob))

phenotypic_category_counts <- same_class_wrong_pred_summary_df %>%
    dplyr::select(Mitocheck_Category, total_count) %>%
    dplyr::group_by(Mitocheck_Category) %>%
    dplyr::summarize(phenotype_category_count = sum(total_count)) %>%
    dplyr::ungroup()

same_class_wrong_pred_summary_df <- same_class_wrong_pred_summary_df %>%
    dplyr::left_join(phenotypic_category_counts, by = "Mitocheck_Category") %>%
    dplyr::mutate(category_proportion = total_count / phenotype_category_count)

same_class_wrong_pred_summary_df$Mitocheck_Category <- factor(
    same_class_wrong_pred_summary_df$Mitocheck_Category,
    levels = rev(levels(same_class_wrong_pred_summary_df$Mitocheck_Category))
)

head(same_class_wrong_pred_summary_df)

correct_class_phenotype_pred_gg <- (
    ggplot(
        same_class_wrong_pred_summary_df,
        aes(
            x = Mitocheck_Category,
            y = category_proportion,
            fill = correct_class_pred
        )
    )
    + geom_bar(stat = "identity")
    + geom_text(
        data = same_class_wrong_pred_summary_df %>%
            dplyr::filter(correct_class_pred == TRUE),
        color = "black",
        aes(label = total_count),
        nudge_y = 0.06
    )
    + coord_flip()
    + scale_fill_manual(
        "Correct\nphenotype\nclass\nprediction?",
        values = focus_corr_colors,
        labels = focus_corr_labels,
        breaks = c("TRUE", "FALSE")
    )
    + theme_bw()
    + phenotypic_ggplot_theme
    + labs(x = "Phenotypic categories", y = "Cell proportions\nof incorrect phenotype predictions")
)

correct_class_phenotype_pred_gg

right_bottom_nested <- (
    correct_pred_proportion_gg / correct_class_phenotype_pred_gg
) + plot_layout(heights = c(1, 0.7)) 

bottom_nested <- (
    per_image_category_gg | right_bottom_nested
) + plot_layout(widths = c(1, 0.9))

compiled_fig <- (
    loio_feature_space_gg /
    bottom_nested
) + plot_annotation(tag_levels = "A") + plot_layout(heights = c(0.3, 1)) 

ggsave(output_fig_loio, dpi = 500, height = 10, width = 12)

compiled_fig

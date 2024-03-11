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

output_fig_loio <- file.path("figures", "main_figure_4_loio.png")

# Set high threshold which represents a more-or-less arbitrary p value cutoff
# for assigning high probability phenotypes to cells
high_threshold <- 0.9

# Set custom labellers for adding context to facet text plotting
custom_labeller <- function(value) {
  paste0("Correct prediction:\n", value)
}

shuffled_labeller <- function(value) {
  paste("Shuffled:", value)
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

#feature_order <- c("CP" = "CP", "DP" = "DP", "CP_and_DP" = "CP_and_DP")

loio_summary_per_phenotype_df$Model_Feature_Type <-
    dplyr::recode_factor(loio_summary_per_phenotype_df$Model_Feature_Type, !!!feature_spaces)

loio_summary_per_phenotype_df$Mitocheck_Plot_Label <-
    dplyr::recode_factor(loio_summary_per_phenotype_df$Mitocheck_Plot_Label, !!!focus_phenotype_labels)

loio_summary_per_phenotype_df$Shuffled <- dplyr::recode_factor(
    loio_summary_per_phenotype_df$Model_type,
    "final" = "FALSE", "shuffled_baseline" = "TRUE"
)

head(loio_summary_per_phenotype_df, 3)

# Focus main result on select LOIO parameters:
# Balanced model, with ic, select feature spaces (CP, DP, CP_and_DP)
loio_summary_per_phenotype_focus_df <- loio_summary_per_phenotype_df %>%
    dplyr::filter(
        Balance_type == "balanced",
        Dataset_type == "ic",
        Model_Feature_Type %in% c("CellProfiler", "DeepProfiler", "CP and DP")
    )

percent_summary_df <- loio_summary_per_phenotype_focus_df %>%
    dplyr::mutate(pass_on_average = Average_Rank < 2) %>%
    dplyr::filter(Shuffled == FALSE) %>%
    dplyr::group_by(
        Mitocheck_Plot_Label,
        Model_Feature_Type
    ) %>%
    dplyr::mutate(total_pass = sum(pass_on_average), total_count = n()) %>%
    dplyr::mutate(
        percent_pass = round((total_pass / total_count) * 100, 1)
    ) %>%
    dplyr::select(
        Mitocheck_Plot_Label,
        Model_Feature_Type,
        percent_pass,
        total_pass,
        total_count
    ) %>%
    dplyr::distinct() %>%
    dplyr::ungroup() %>%
    dplyr::mutate(add_plot_text = paste0(total_pass, "/", total_count, "\n", "(", percent_pass, "%)"))

head(percent_summary_df)

per_image_category_gg <- (
    ggplot(loio_summary_per_phenotype_focus_df, aes(x = Average_Rank, y = Average_P_Value))
    + geom_point(aes(size = Count, color = Shuffled), alpha = 0.2)
    + theme_bw()
    + phenotypic_ggplot_theme
    + facet_grid(
        "Mitocheck_Plot_Label~Model_Feature_Type"
    )
    + geom_text(
        data=percent_summary_df,
        aes(label = add_plot_text, x = 12, y = 0.7),
        size = 5
    )
    + labs(
        x = "Average rank of correct label\n(per held out image)",
        y = "Average probability of correct label\n(per held out image)"
    )

    + scale_size_continuous(
        name = "Per held\nout image\ncell count"
    )
    + scale_color_manual(
        "Is data\nrandomly\nshuffled?",
        values = shuffled_colors
    )
    + geom_vline(xintercept=2, linetype = "dashed", color = "red")
    + theme(
        strip.text = element_text(size = 11),
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

phenotypic_class_category_counts <- loio_focus_df %>%
    dplyr::ungroup() %>%
    dplyr::select(
        Mitocheck_Phenotypic_Class,
        Model_type,
        Balance_type,
        Model_Feature_Type,
        correct_pred,
        correct_class_pred
    ) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_type,
        Balance_type,
        Model_Feature_Type,
        correct_pred,
        correct_class_pred
    ) %>%
    dplyr::summarize(phenotype_count = n()) %>%
    dplyr::ungroup()

loio_thresh_df <- loio_focus_df %>%
    dplyr::ungroup() %>%
    dplyr::mutate(pass_threshold = paste(Predicted_Probability >= high_threshold)) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_type,
        Balance_type,
        Model_Feature_Type,
        correct_pred,
        correct_class_pred,
        pass_threshold
    ) %>%
    dplyr::summarize(count = n()) %>%
    dplyr::left_join(
        phenotypic_class_category_counts,
        by = c(
            "Mitocheck_Phenotypic_Class",
            "Model_type",
            "Balance_type",
            "Model_Feature_Type",
            "correct_pred"
        )
    ) %>%
    dplyr::mutate(phenotype_prop = count / phenotype_count)

phenotypic_class_counts <- loio_focus_df %>%
    dplyr::ungroup() %>%
    dplyr::select(
        Mitocheck_Phenotypic_Class,
        Model_type,
        Balance_type,
        Model_Feature_Type,
        correct_pred
    ) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_type,
        Balance_type,
        Model_Feature_Type,
        correct_pred
    ) %>%
    dplyr::summarize(phenotype_count = n()) %>%
    dplyr::ungroup()

loio_thresh_df <- loio_focus_df %>%
    dplyr::mutate(pass_threshold = paste(Predicted_Probability >= high_threshold)) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_type,
        Balance_type,
        Model_Feature_Type,
        correct_pred,
        pass_threshold
    ) %>%
    dplyr::summarize(count = n()) %>%
    dplyr::left_join(
        phenotypic_class_counts,
        by = c(
            "Mitocheck_Phenotypic_Class",
            "Model_type",
            "Balance_type",
            "Model_Feature_Type",
            "correct_pred"
        )
    ) %>%
    dplyr::mutate(phenotype_prop = count / phenotype_count)

# Reverse order of predicted label for plotting
loio_thresh_df$Mitocheck_Phenotypic_Class <-
    factor(loio_thresh_df$Mitocheck_Phenotypic_Class, levels = rev(unique(loio_thresh_df$Mitocheck_Phenotypic_Class)))

loio_thresh_df$Shuffled <- dplyr::recode_factor(
    loio_thresh_df$Model_type,
    "final" = "FALSE", "shuffled_baseline" = "TRUE"
)

head(loio_thresh_df)

correct_pred_proportion_gg <- (
    ggplot(
        loio_thresh_df %>%
            dplyr::filter(
                Balance_type == "balanced",
                Shuffled == FALSE
            ),
        aes(
            x = phenotype_prop,
            y = Mitocheck_Phenotypic_Class,
            fill = pass_threshold
        )
    )
    + geom_bar(stat = "identity")
    + geom_text(
        data = loio_thresh_df %>%
            dplyr::filter(
                Balance_type == "balanced",
                Shuffled == FALSE,
                pass_threshold == TRUE
            ),
        color = "black",
        aes(label = count),
        nudge_x = 0.07,
        size = 4
    )
    + facet_grid(
        "Model_Feature_Type~correct_pred",
        labeller = labeller(correct_pred = custom_labeller, Shuffled = shuffled_labeller)
    )
    + theme_bw()
    + phenotypic_ggplot_theme
    + theme(
        axis.text = element_text(size = 9.5),
        strip.text = element_text(size = 11)
    )
    + scale_fill_manual(
        paste0("Does cell\npass strict\nthreshold?\n(p = ", high_threshold, ")"),
        values = focus_corr_colors,
        labels = focus_corr_labels,
        breaks = c("TRUE", "FALSE")
    )
    + labs(x = "Cell proportions", y = "Mitocheck phenotypes")
)

correct_pred_proportion_gg

same_class_wrong_pred_summary_df <- loio_focus_df %>%
    dplyr::filter(correct_pred == "FALSE") %>%
    dplyr::filter(rank_value == 1) %>%
    dplyr::group_by(
        Mitocheck_Phenotypic_Class,
        Model_type,
        Balance_type,
        Model_Feature_Type,
        correct_class_pred
    ) %>%
    dplyr::summarize(count = n(), avg_prob = mean(Predicted_Probability)) %>%
    dplyr::ungroup() %>%
    dplyr::left_join(phenotype_categories_df, by = "Mitocheck_Phenotypic_Class") %>%
    dplyr::group_by(
        Mitocheck_Category,
        Model_type,
        Balance_type,
        Model_Feature_Type,
        correct_class_pred
    ) %>%
    dplyr::summarize(total_count = sum(count), avg_prob = mean(avg_prob))

phenotypic_category_counts <- same_class_wrong_pred_summary_df %>%
    dplyr::select(
        Mitocheck_Category,
        Model_type,
        Balance_type,
        Model_Feature_Type,
        total_count
    ) %>%
    dplyr::group_by(
        Mitocheck_Category,
        Model_type,
        Balance_type,
        Model_Feature_Type,
    ) %>%
    dplyr::summarize(phenotype_category_count = sum(total_count)) %>%
    dplyr::ungroup()

same_class_wrong_pred_summary_df <- same_class_wrong_pred_summary_df %>%
    dplyr::left_join(
        phenotypic_category_counts,
        by = c(
            "Mitocheck_Category",
            "Model_type",
            "Balance_type",
            "Model_Feature_Type"
        )
    ) %>%
    dplyr::mutate(category_proportion = total_count / phenotype_category_count)

same_class_wrong_pred_summary_df$Mitocheck_Category <- factor(
    same_class_wrong_pred_summary_df$Mitocheck_Category,
    levels = rev(levels(same_class_wrong_pred_summary_df$Mitocheck_Category))
)

same_class_wrong_pred_summary_df$Shuffled <- dplyr::recode_factor(
    same_class_wrong_pred_summary_df$Model_type,
    "final" = "FALSE", "shuffled_baseline" = "TRUE"
)

head(same_class_wrong_pred_summary_df)

correct_class_phenotype_pred_gg <- (
    ggplot(
        same_class_wrong_pred_summary_df %>%
        dplyr::filter(
                Balance_type == "balanced",
                Shuffled == FALSE
        ),
        aes(
            x = Mitocheck_Category,
            y = category_proportion,
            fill = correct_class_pred
        )
    )
    + geom_bar(stat = "identity")
    + facet_grid(
        "~Model_Feature_Type",
        labeller = labeller(Shuffled = shuffled_labeller)
    )
    + geom_text(
        data = same_class_wrong_pred_summary_df %>%
            dplyr::filter(
                correct_class_pred == TRUE,
                Balance_type == "balanced",
                Shuffled == FALSE
            ),
        color = "black",
        aes(label = total_count),
        nudge_y = 0.12,
        size = 4
    )
    + coord_flip()
    + scale_fill_manual(
        "Correct\nphenotype\ncategory\nprediction?",
        values = focus_corr_colors,
        labels = focus_corr_labels,
        breaks = c("TRUE", "FALSE")
    )
    + theme_bw()
    + phenotypic_ggplot_theme
    + theme(
        axis.text.x = element_text(size = 7.5),
        strip.text = element_text(size = 11)
    )
    + labs(x = "Phenotype categories", y = "Cell proportions")
)

correct_class_phenotype_pred_gg

right_bottom_nested <- (
    correct_pred_proportion_gg / correct_class_phenotype_pred_gg
) + plot_layout(heights = c(1, 0.2)) 

compiled_fig <- (
    per_image_category_gg | right_bottom_nested
) + plot_layout(widths = c(1, 0.77)) + plot_annotation(tag_levels = "A")

ggsave(output_fig_loio, dpi = 500, height = 10, width = 15)

compiled_fig

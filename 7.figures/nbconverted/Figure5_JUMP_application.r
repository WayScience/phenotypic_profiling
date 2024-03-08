suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))

# Load variables important for plotting (e.g., themes, phenotypes, etc.)
source("themes.r")

# Set output file paths
output_folder <- "figures"

jump_phenotype_enrichment_supplementary_file <- file.path(output_folder, "supplementary", "supplementary_jump_enrichment_shuffled_and_extended.png")
output_fig_5_file <- file.path(output_folder, "main_figure_5_jump_application.png")

# Set paths to load UMAP coordinates
# Loaded from: https://github.com/WayScience/JUMP-single-cell/tree/main/3.analyze_data/UMAP_analysis/results/Mito_JUMP_areashape_features
repo <- "https://github.com/WayScience/JUMP-single-cell"
commit_hash <- "d95bcc99cb99fe9093bf0d73e2019edeb1e57e87"
umap_file_path <- "3.analyze_data/UMAP_analysis/results/"

all_feature_umap_file <- "Mito_JUMP_all_features/Mito_JUMP_all_features_final_greg_areashape_model.tsv"
area_shape_umap_file <- "Mito_JUMP_areashape_features/Mito_JUMP_areashape_features_final_all_features_model.tsv"

all_feature_umap <- paste(repo, "raw", commit_hash, umap_file_path, all_feature_umap_file, sep = "/")
area_shape_umap <- paste(repo, "raw", commit_hash, umap_file_path, area_shape_umap_file, sep = "/")

# Load UMAP coordinates
all_feature_umap_df <- readr::read_tsv(
    all_feature_umap,
    col_types = readr::cols(
        .default = "d",
        Metadata_data_name = "c",
        Metadata_Predicted_Class = "c"
    )
)

area_shape_umap_df <- readr::read_tsv(
    area_shape_umap,
    col_types = readr::cols(
        .default = "d",
        Metadata_data_name = "c",
        Metadata_Predicted_Class = "c"
    )
)

print(dim(area_shape_umap_df))
head(area_shape_umap_df)

all_feature_umap_gg <- (
    ggplot(all_feature_umap_df, aes(x = UMAP0, y = UMAP1))
    + geom_point(
        aes(color = Metadata_data_name),
        size = 0.1,
        alpha = 0.3,
        shape = 16
    )
    + geom_point(
        data = all_feature_umap_df %>% dplyr::filter(Metadata_data_name == "mitocheck"),
        aes(color = Metadata_data_name),
        size = 0.4,
        alpha = 0.3,
        shape = 16
    )
    + theme_bw()
    + phenotypic_ggplot_theme
    + guides(
        color = guide_legend(
            override.aes = list(size = 2)
        )
    )
    + labs(x = "UMAP 1", y = "UMAP 2")
    + scale_color_manual(
        "Dataset",
        values = dataset_colors,
        labels = dataset_labels
    )
    + ggtitle("All nuclei features")
    + coord_fixed()
)

all_feature_umap_gg

area_shape_umap_gg <- (
    ggplot(area_shape_umap_df, aes(x = UMAP0, y = UMAP1))
    + geom_point(
        aes(color = Metadata_data_name),
        size = 0.1,
        alpha = 0.3,
        shape = 16
    )
    + geom_point(
        data = area_shape_umap_df %>% dplyr::filter(Metadata_data_name == "mitocheck"),
        aes(color = Metadata_data_name),
        size = 0.4,
        alpha = 0.3,
        shape = 16
    )
    + theme_bw()
    + phenotypic_ggplot_theme
    + guides(
        color = guide_legend(
            override.aes = list(size = 2)
        )
    )
    + labs(x = "UMAP 1", y = "UMAP 2")
    + scale_color_manual(
        "Dataset",
        values = dataset_colors,
        labels = dataset_labels
    )
    + ggtitle("Nuclei AreaShape only")
    + coord_fixed()
) 

area_shape_umap_gg

# Set file paths
jump_path <- file.path("..", "3.evaluate_model", "jump_phenotype_profiles")

jump_compare_conditions_file <- file.path(jump_path, "jump_compare_cell_types_and_time_across_phenotypes.tsv.gz")
jump_phenotype_umap_file <- file.path(jump_path, "jump_phenotype_profiling_umap.tsv.gz")

#  Load and process data
jump_compare_df <- readr::read_tsv(
    jump_compare_conditions_file,
    show_col_types = FALSE
) %>%
    # Generate new columns that we will use for plotting:
    # 1) Phenotype colors
    # Note, we define focus_phenotypes in themes.r
    dplyr::mutate(phenotype_plot_label = if_else(
        phenotype %in% focus_phenotypes,
        phenotype,
        "Other"
    )) %>%
    # 2) High vs. low incubation time
    dplyr::mutate(time_plot_label = if_else(
        treatment_type == "orf" & Time == 48,
        "Low",
        "tbd"
    )) %>%
    dplyr::mutate(time_plot_label = if_else(
        treatment_type == "orf" & Time == 96,
        "High",
        time_plot_label
    )) %>%
    dplyr::mutate(time_plot_label = if_else(
        treatment_type == "compound" & Time == 24,
        "Low",
        time_plot_label
    )) %>%
    dplyr::mutate(time_plot_label = if_else(
        treatment_type == "compound" & Time == 48,
        "High",
        time_plot_label
    )) %>%
    dplyr::mutate(time_plot_label = if_else(
        treatment_type == "crispr" & Time == 96,
        "Low",
        time_plot_label
    )) %>%
    dplyr::mutate(time_plot_label = if_else(
        treatment_type == "crispr" & Time == 144,
        "High",
        time_plot_label
    )) %>%
    dplyr::mutate(
        neg_log10_pval_A549 = -log10(p_value_A549),
        neg_log10_pval_U2OS = -log10(p_value_U2OS)
    )

jump_compare_df$time_plot_label <-
    factor(jump_compare_df$time_plot_label, levels = c("Low", "High"))

jump_compare_df$phenotype_plot_label <-
    dplyr::recode_factor(jump_compare_df$phenotype_plot_label, !!!focus_phenotype_labels)

print(dim(jump_compare_df))
head(jump_compare_df)

custom_time_labeller <- function(value) {
  paste("Time:", value)
}

compare_phenotype_enrichment_ggs <- list()
for (model_type in c("final", "shuffled")) {
    jump_subset_compare_df <- jump_compare_df %>%
        dplyr::filter(Metadata_model_type == !!model_type)
    
    compare_phenotype_enrichment_ggs[[model_type]] <- (
        ggplot(
            jump_subset_compare_df,
            aes(
                x = neg_log10_pval_A549,
                y = neg_log10_pval_U2OS,
                color = phenotype_plot_label
            )
        )
        + geom_point(
            size = 2.1,
            alpha = 0.4,
            shape = 16
        )
        + geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed")
        + theme_bw()
        + scale_color_manual(
            "Phenotype",
            values = focus_phenotype_colors,
            labels = focus_phenotype_labels
        )
        + phenotypic_ggplot_theme
        + guides(
            color = guide_legend(
                override.aes = list(size = 2)
            )
        )
        + facet_grid(
            "time_plot_label~treatment_type",
            labeller = labeller(time_plot_label = custom_time_labeller)
        )
        + coord_fixed()
        + scale_x_continuous(limits = c(0, 130))
        + scale_y_continuous(limits = c(0, 130))
        + labs(
            x = "A549 phenotype enrichment\n(KS test -log10 pvalue)",
            y = "U20S phenotype enrichment\n(KS test -log10 pvalue)"
        )
    )
}


compare_phenotype_enrichment_ggs[["final"]]

top_plot <- (
    all_feature_umap_gg | 
    area_shape_umap_gg
) + plot_layout(guides = "collect")

fig_5_gg <- (
    top_plot /
    compare_phenotype_enrichment_ggs[["final"]]
) + plot_annotation(tag_levels = "A") + plot_layout(heights = c(1, 1.4))

ggsave(output_fig_5_file, dpi = 500, height = 8.5, width = 8)

fig_5_gg

jump_other_phenotype_df <- jump_compare_df %>% 
    dplyr::filter(
        !phenotype %in% focus_phenotypes,
        Metadata_model_type == "final",
        treatment_type == "compound"
    )

head(jump_other_phenotype_df)

# Custom function for name repair
name_repair_function <- function(names) {
  names[4] <- paste0(names[4], "_original")
  return(names)
}

expanded_phenotype_colors_time_ggs <- list()
focus_phenotype_colors_time_ggs <- list()

for (time_point in c("Low", "High")) {
    # Focus other phenotypes to specific time points
    jump_other_phenotype_per_time_df <- jump_other_phenotype_df %>%
        dplyr::filter(time_plot_label == !!time_point)

    # Create a background data for plotting gray points
    df_background <- tidyr::crossing(
        jump_other_phenotype_per_time_df,
        phenotype = unique(jump_other_phenotype_per_time_df$phenotype),
        .name_repair = name_repair_function
    )

    # Create the figure for other phenotypes
    jump_sup_fig_other_gg <- (
        ggplot(
            jump_other_phenotype_per_time_df,
            aes(x = neg_log10_pval_A549, y = neg_log10_pval_U2OS)
        )
        + geom_point(
            data = df_background,
            color = "lightgray",
            size = 0.8,
            alpha = 0.6
        )
        + geom_point(
            aes(color = phenotype),
            size = 0.8
        )
        + geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed")
        + facet_grid("treatment_type~phenotype")
        + theme_bw()
        + phenotypic_ggplot_theme
        + guides(
            color = guide_legend(
                override.aes = list(size = 2)
            )
        )
        + labs(
            x = "A549 enrichment\n(KS test -log10 pvalue)",
            y = "U20S enrichment\n(KS test -log10 pvalue)"
        )
        + theme(
            legend.position = "none",
            strip.text = element_text(size = 8),
        )
        + ggtitle(paste("Time:", time_point))
    )

    # Save in gg list
    expanded_phenotype_colors_time_ggs[[time_point]] <- jump_sup_fig_other_gg

    # Now switch to the focus phenotypes
    jump_focus_phenotype_per_time_df <- jump_compare_df %>% 
        dplyr::filter(
            phenotype %in% focus_phenotypes,
            Metadata_model_type == "final",
            time_plot_label == !!time_point,
            treatment_type == "compound"
        )

    df_background <- tidyr::crossing(
        jump_focus_phenotype_per_time_df,
        phenotype = unique(jump_focus_phenotype_per_time_df$phenotype),
        .name_repair = name_repair_function
    )
    
    # Create the figure for focus phenotypes
    jump_sup_fig_focus_gg <- (
        ggplot(
            jump_focus_phenotype_per_time_df,
            aes(x = neg_log10_pval_A549, y = neg_log10_pval_U2OS)
        )
        + geom_point(
            data = df_background,
            color = "lightgray",
            size = 0.8,
            alpha = 0.6
        )
        + geom_point(
            aes(color = phenotype),
            size = 0.8
        )
        + geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed")
        + facet_grid("treatment_type~phenotype")
        + theme_bw()
        + phenotypic_ggplot_theme
        + guides(
            color = guide_legend(
                override.aes = list(size = 2)
            )
        )
        + scale_color_manual(
            "Phenotype",
            values = focus_phenotype_colors,
            labels = focus_phenotype_labels
        )
        + labs(
            x = "A549 enrichment\n(KS test -log10 pvalue)",
            y = "U20S enrichment\n(KS test -log10 pvalue)"
        )
        + theme(
            legend.position = "none",
            strip.text = element_text(size = 8.5),
        )
        + ggtitle(paste("Time:", time_point))
    )

    # Save in gg list
    focus_phenotype_colors_time_ggs[[time_point]] <- jump_sup_fig_focus_gg
}

## Save supplementary figure for shuffled p values
top_plot <- (
   compare_phenotype_enrichment_ggs[["shuffled"]] + ggtitle("Shuffled data results") | plot_spacer()
) + plot_layout(widths = c(1, 1))

nested_plot <- (
    focus_phenotype_colors_time_ggs[["High"]] | plot_spacer()
) + plot_layout(widths = c(3, 1.35))

jump_kstest_full_high_fig <- (
    nested_plot / expanded_phenotype_colors_time_ggs[["High"]]
) + plot_layout(heights = c(1, 1))

nested_low_plot <- (
    focus_phenotype_colors_time_ggs[["Low"]] | plot_spacer()
) + plot_layout(widths = c(3, 1.35))

jump_kstest_full_low_fig <- (
    nested_low_plot / expanded_phenotype_colors_time_ggs[["Low"]]
) + plot_layout(heights = c(1, 1))


jump_phenotype_enrichment_supplementary_gg <- (
    top_plot /
    jump_kstest_full_low_fig /
    jump_kstest_full_high_fig
    ) + plot_layout(nrow = 3, heights = c(1, 1, 1)) + plot_annotation(tag_levels = list(c("A", "B", "", "C", "")))

ggsave(jump_phenotype_enrichment_supplementary_file, dpi = 500, height = 13, width = 13)

jump_phenotype_enrichment_supplementary_gg

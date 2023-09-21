suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
source("themes.r")

figure_dir <- "figures"
output_main_figure_2 <- file.path(figure_dir, "main_figure_2_umap_and_correlation.png")
output_sup_fig_corr <- file.path(figure_dir, "supplementary", "supplementary_pairwise_correlations.png")

focus_corr_colors = c(
    "TRUE" = "blue",
    "FALSE" = "orange"
)
focus_corr_labels  = c(
    "TRUE" = "Yes",
    "FALSE" = "No"
)

# Load UMAP coordinates and process
umap_dir <- file.path("../../mitocheck_data/4.analyze_data/results/")
umap_file <- file.path(umap_dir, "compiled_2D_umap_embeddings.csv")

umap_df <- readr::read_csv(
    umap_file,
    col_types = readr::cols(
        .default = "c",
        Embedding_Value = "d"
    )
) %>%
    dplyr::select(!...1) %>%#7570b3
    tidyr::pivot_wider(names_from = UMAP_Embedding, values_from = Embedding_Value) %>%
    dplyr::mutate(Mitocheck_Plot_Label = if_else(
        Mitocheck_Phenotypic_Class %in% focus_phenotypes,
        Mitocheck_Phenotypic_Class,
        "Other"
    ))

umap_df$Mitocheck_Plot_Label <-
    dplyr::recode_factor(umap_df$Mitocheck_Plot_Label, !!!focus_phenotype_labels)

umap_df$Feature_Type <-
    dplyr::recode_factor(umap_df$Feature_Type, !!!facet_labels)

head(umap_df)

umap_fig_gg <- (
    ggplot(umap_df, aes(x = UMAP1, y = UMAP2))
    + geom_point(
        aes(color = Mitocheck_Plot_Label),
        size = 0.05,
        alpha = 0.4
    )
    + facet_grid("~Feature_Type")
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
    + labs(x = "UMAP 1", y = "UMAP 2")
)

umap_fig_gg

# Load data
corr_dir <- file.path("..", "1.split_data", "data")

corr_df <- list()
for (feature_space in c("CP", "DP", "CP_and_DP")) {
    file_name <- file.path(
        corr_dir,
        paste0("pairwise_correlations_", feature_space, ".tsv.gz")
    )
    corr_df[[feature_space]] <- (
        readr::read_tsv(file_name, show_col_types = FALSE) %>%
        dplyr::mutate(feature_space = feature_space)
    )
}

corr_df <- do.call(rbind, corr_df) %>%
    dplyr::mutate(same_label = Row_Label == Pairwise_Row_Label)

corr_df$feature_space <-
    dplyr::recode_factor(corr_df$feature_space, !!!facet_labels)

print(dim(corr_df))
head(corr_df)

focused_corr_df <- corr_df %>%
    dplyr::filter(
        Row_Label %in% focus_phenotypes |
        Pairwise_Row_Label %in% focus_phenotypes
    )

print(dim(focused_corr_df))
head(focused_corr_df)

all_focus_phenotype_corr_df <- list()

for (phenotype in focus_phenotypes) {
    phenotype_specific_corr_df <- focused_corr_df %>%
        dplyr::filter(
            Row_Label == !!phenotype |
            Pairwise_Row_Label == !!phenotype
        )
    
    same_phenotype_specific_corr_df <- phenotype_specific_corr_df %>%
        dplyr::filter(same_label) %>%
        dplyr::mutate(first_compare = phenotype, second_compare = phenotype)
    
    diff_phenotype_specific_corr_df <- phenotype_specific_corr_df %>%
        dplyr::filter(!same_label) %>%
        dplyr::mutate(first_compare = !!phenotype) %>%
        dplyr::mutate(
            second_compare = if_else(
                Row_Label == !!phenotype,
                Pairwise_Row_Label,
                Row_Label
            )
        )

    all_focus_phenotype_corr_df[[phenotype]] <- dplyr::bind_rows(
        same_phenotype_specific_corr_df,
        diff_phenotype_specific_corr_df
    )
}

all_focus_phenotype_corr_df <- do.call(rbind, all_focus_phenotype_corr_df) 

all_focus_phenotype_corr_df$first_compare <-
    dplyr::recode_factor(all_focus_phenotype_corr_df$first_compare, !!!focus_phenotype_labels)

dim(all_focus_phenotype_corr_df)
head(all_focus_phenotype_corr_df, 3)

focus_phenotype_gg <- (
    ggplot(all_focus_phenotype_corr_df, aes(x = Correlation))
    + geom_density(aes(fill = same_label), alpha = 0.5)
    + facet_grid("feature_space~first_compare")
    + theme_bw()
    + scale_fill_manual(
        "Are cell\npairs of\nthe same\nphenotype?",
        values = focus_corr_colors,
        labels = focus_corr_labels
    )
    + phenotypic_ggplot_theme
    + guides(
        color = guide_legend(
            override.aes = list(size = 2)
        )
    )
    + labs(x = "pairwise Pearson correlation", y = "Density")
    + geom_vline(xintercept = 0, linetype = "dashed", color = "darkgrey")
)

focus_phenotype_gg

fig_2_gg <- (
    umap_fig_gg /
    focus_phenotype_gg
) + plot_annotation(tag_levels = "A") + plot_layout(heights = c(0.6, 1))

ggsave(output_main_figure_2, dpi = 500, height = 7, width = 8)

fig_2_gg

total_pairwise_corr_gg <- (
    ggplot(corr_df, aes(x = Correlation))
    + geom_density(aes(fill = same_label), alpha = 0.5)
    + facet_wrap("~feature_space")
    + theme_bw()
    + scale_fill_manual(
        "Are cell\npairs of\nthe same\nphenotype?",
        values = focus_corr_colors,
        labels = focus_corr_labels
    )
    + phenotypic_ggplot_theme
    + guides(
        color = guide_legend(
            override.aes = list(size = 2)
        )
    )
    + labs(x = "pairwise Pearson correlation", y = "Density")
    + geom_vline(xintercept = 0, linetype = "dashed", color = "darkgrey")
)

total_pairwise_corr_gg

other_phenotypes <- setdiff(unique(corr_df$Row_Label), focus_phenotypes)
other_phenotypes

other_corr_df <- corr_df %>%
    dplyr::filter(
        Row_Label %in% other_phenotypes |
        Pairwise_Row_Label %in% other_phenotypes
    )

print(dim(other_corr_df))
head(other_corr_df)

other_phenotype_corr_df <- list()

for (phenotype in other_phenotypes) {
    phenotype_specific_corr_df <- other_corr_df %>%
        dplyr::filter(
            Row_Label == !!phenotype |
            Pairwise_Row_Label == !!phenotype
        )
    
    same_phenotype_specific_corr_df <- phenotype_specific_corr_df %>%
        dplyr::filter(same_label) %>%
        dplyr::mutate(first_compare = phenotype, second_compare = phenotype)
    
    diff_phenotype_specific_corr_df <- phenotype_specific_corr_df %>%
        dplyr::filter(!same_label) %>%
        dplyr::mutate(first_compare = !!phenotype) %>%
        dplyr::mutate(
            second_compare = if_else(
                Row_Label == !!phenotype,
                Pairwise_Row_Label,
                Row_Label
            )
        )

    other_phenotype_corr_df[[phenotype]] <- dplyr::bind_rows(
        same_phenotype_specific_corr_df,
        diff_phenotype_specific_corr_df
    )
}

other_phenotype_corr_df <- do.call(rbind, other_phenotype_corr_df) 

dim(other_phenotype_corr_df)
head(other_phenotype_corr_df, 3)

other_phenotype_gg <- (
    ggplot(other_phenotype_corr_df, aes(x = Correlation))
    + geom_density(aes(fill = same_label), alpha = 0.5)
    + facet_grid("feature_space~first_compare")
    + theme_bw()
    + scale_fill_manual(
        "Are cell\npairs of\nthe same\nphenotype?",
        values = focus_corr_colors,
        labels = focus_corr_labels
    )
    + phenotypic_ggplot_theme
    + guides(
        color = guide_legend(
            override.aes = list(size = 2)
        )
    )
    + labs(x = "pairwise Pearson correlation", y = "Density")
    + geom_vline(xintercept = 0, linetype = "dashed", color = "darkgrey")
    + theme(legend.position = "none")
)

other_phenotype_gg

nested_plot <- (
    total_pairwise_corr_gg | plot_spacer()
) + plot_layout(widths = c(3, 2))

sup_fig_gg <- (
    nested_plot /
    other_phenotype_gg
) + plot_annotation(tag_levels = "A") + plot_layout(heights = c(0.3, 1))

ggsave(output_sup_fig_corr, dpi = 500, height = 7, width = 14.5)

sup_fig_gg

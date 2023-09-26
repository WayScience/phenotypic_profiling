suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
source("themes.r")

figure_dir <- "figures"
sup_figure_dir <- file.path(figure_dir, "supplementary")

output_supplementary_umap_figure <- file.path(sup_figure_dir, "supplementary_umap.png")

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
    dplyr::select(!...1) %>%
    dplyr::filter(Mitocheck_Phenotypic_Class != "Folded") %>%  # Drop folded class due to low n
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

umap_other_df <- umap_df %>% dplyr::filter(!Mitocheck_Phenotypic_Class %in% focus_phenotypes)

# Custom function for name repair
name_repair_function <- function(names) {
  names[1] <- paste0(names[1], "_original")
  return(names)
}

df_background <- tidyr::crossing(
    umap_df,
    Mitocheck_Phenotypic_Class = unique(umap_other_df$Mitocheck_Phenotypic_Class),
    .name_repair = name_repair_function
)

umap_sup_fig_other_gg <- (
    ggplot(
        umap_other_df,
        aes(x = UMAP1, y = UMAP2)
    )
    + geom_point(
        data = df_background,
        color = "lightgray",
        size = 0.1,
        alpha = 0.4
    )
    + geom_point(
        aes(color = Mitocheck_Phenotypic_Class),
        size = 0.1
    )
    + facet_grid("Feature_Type~Mitocheck_Phenotypic_Class")
    + theme_bw()
    + phenotypic_ggplot_theme
    + guides(
        color = guide_legend(
            override.aes = list(size = 2)
        )
    )
    + labs(x = "UMAP 1", y = "UMAP 2")
    + theme(
        legend.position = "none",
        strip.text = element_text(size = 8.5),
    )
)

umap_sup_fig_other_gg

umap_focus_df <- umap_df %>% dplyr::filter(Mitocheck_Phenotypic_Class %in% focus_phenotypes)

df_background <- tidyr::crossing(
    umap_df,
    Mitocheck_Phenotypic_Class = unique(umap_focus_df$Mitocheck_Phenotypic_Class),
    .name_repair = name_repair_function
)

umap_sup_fig_focus_gg <- (
    ggplot(
        umap_df %>% dplyr::filter(Mitocheck_Plot_Label %in% focus_phenotypes),
        aes(x = UMAP1, y = UMAP2)
    )
    + geom_point(
        data = df_background,
        color = "lightgray",
        size = 0.1,
        alpha = 0.4
    )
    + geom_point(
        aes(color = Mitocheck_Plot_Label),
        size = 0.1
    )

    + facet_grid("Feature_Type~Mitocheck_Phenotypic_Class")
    + theme_bw()
    + phenotypic_ggplot_theme
    + guides(
        color = guide_legend(
            override.aes = list(size = 2)
        )
    )
    + labs(x = "UMAP 1", y = "UMAP 2")
    + scale_color_manual(
        "Phenotype",
        values = focus_phenotype_colors,
        labels = focus_phenotype_labels
    )
    + theme(legend.position = "none")
)

umap_sup_fig_focus_gg

nested_plot <- (
    umap_sup_fig_focus_gg | plot_spacer()
) + plot_layout(widths = c(3, 1.35))

umap_full_sup_fig <- (
    nested_plot / umap_sup_fig_other_gg
) + plot_layout(heights = c(1, 1))

ggsave(output_supplementary_umap_figure, dpi = 500, height = 10, width = 13)

umap_full_sup_fig

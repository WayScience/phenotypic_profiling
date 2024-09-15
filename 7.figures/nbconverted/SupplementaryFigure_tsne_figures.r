suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))

# Load variables important for plotting (e.g., themes, phenotypes, etc.)
source("themes.r")

figure_dir <- "figures"

output_sup_fig_tsne <- file.path(
    figure_dir,
    "supplementary",
    "supplementary_tsne_figure.png"
)

focus_corr_colors = c(
    "TRUE" = "blue",
    "FALSE" = "orange"
)
focus_corr_labels  = c(
    "TRUE" = "Yes",
    "FALSE" = "No"
)

list_of_perplexities <- c(2, 10, 15, 30, 40, 60, 80, 100, 150, 300)

# Load tsne coordinates and process
tsne_dir <- file.path("../3.evaluate_model/evaluations/")
tsne_file <- file.path(tsne_dir, "tsne_embeddings.csv.gz")

tsne_df <- readr::read_tsv(
    tsne_file,
    col_types = readr::cols(
        .default = "c",
        tsne_x = "d",
        tsne_y = "d",
        perplexity = "d"
    )
) %>%
    # Note, we define focus_phenotypes in themes.r
    dplyr::mutate(Mitocheck_Plot_Label = if_else(
        Mitocheck_Phenotypic_Class %in% focus_phenotypes,
        Mitocheck_Phenotypic_Class,
        "Other"
    ))

tsne_df$Mitocheck_Plot_Label <-
    dplyr::recode_factor(tsne_df$Mitocheck_Plot_Label, !!!focus_phenotype_labels)

tsne_df$Feature_Type <-
    dplyr::recode_factor(tsne_df$feature_group, !!!facet_labels)

# Select certain columns
tsne_df <- tsne_df %>%
    dplyr::select(
        Mitocheck_Phenotypic_Class,
        Feature_Type,
        Cell_UUID,
        tsne_x,
        tsne_y,
        Mitocheck_Plot_Label,
        perplexity
    )

print(dim(tsne_df))
head(tsne_df)

for(perplexity in list_of_perplexities){
    output_tsne_figure_file <- file.path(
        "figures",
        "tsne",
        paste0("tsne_figure_perplexity_", perplexity, ".png")
    )

    # Create a background dataset to show in greyed color across all facets
    tsne_focus_df <- tsne_df %>%
        dplyr::filter(perplexity == !!perplexity) %>%
        dplyr::filter(Mitocheck_Phenotypic_Class %in% focus_phenotypes)
    
    tsne_other_df <- tsne_df %>%
        dplyr::filter(perplexity == !!perplexity) %>%
        dplyr::filter(!Mitocheck_Phenotypic_Class %in% focus_phenotypes)
    
    # Custom function for name repair
    name_repair_function <- function(names) {
      names[1] <- paste0(names[1], "_original")
      return(names)
    }
    
    df_focus_background <- tidyr::crossing(
        tsne_df %>% dplyr::filter(perplexity == !!perplexity),
        Mitocheck_Phenotypic_Class = unique(tsne_focus_df$Mitocheck_Phenotypic_Class),
        .name_repair = name_repair_function
    )
    
    df_background <- tidyr::crossing(
        tsne_df %>% dplyr::filter(perplexity == !!perplexity),
        Mitocheck_Phenotypic_Class = unique(tsne_other_df$Mitocheck_Phenotypic_Class),
        .name_repair = name_repair_function
    )
    
    tsne_fig_gg <- (
        ggplot(
            tsne_focus_df,
            aes(x = tsne_x, y = tsne_y)
        )
        + geom_point(
            data = df_focus_background,
            color = "lightgray",
            size = 0.05,
            alpha = 0.4
        )
        + geom_point(
            aes(color = Mitocheck_Phenotypic_Class),
            size = 0.05
        )
        + facet_grid("Feature_Type~Mitocheck_Phenotypic_Class")
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
        + labs(x = "tSNE 1", y = "tSNE 2")
    )
    
    tsne_fig_other_gg <- (
        ggplot(
            tsne_other_df,
            aes(x = tsne_x, y = tsne_y)
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
        + labs(x = "tSNE 1", y = "tSNE 2")
        + theme(
            legend.position = "none",
            strip.text = element_text(size = 8.5),
        )
    )
    
    nested_plot <- (
        tsne_fig_gg | plot_spacer()
    ) + plot_layout(widths = c(3, 1.35))
    
    tsne_full_sup_fig <- (
        nested_plot / tsne_fig_other_gg
    ) + plot_layout(heights = c(1, 1))
    
    ggsave(output_tsne_figure_file, dpi = 500, height = 10, width = 13)
    
    tsne_full_sup_fig
}

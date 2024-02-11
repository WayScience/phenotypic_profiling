suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))

# Load variables important for plotting (e.g., themes, phenotypes, etc.)
source("themes.r")

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
        alpha = 0.5
    )
    + geom_point(
        data = all_feature_umap_df %>% dplyr::filter(Metadata_data_name == "mitocheck"),
        aes(color = Metadata_data_name),
        size = 0.5,
        alpha = 0.5
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
        "All features\n\nDataset",
        values = dataset_colors,
        labels = dataset_labels
    )
)

all_feature_umap_gg

area_shape_umap_gg <- (
    ggplot(area_shape_umap_df, aes(x = UMAP0, y = UMAP1))
    + geom_point(
        aes(color = Metadata_data_name),
        size = 0.07,
        alpha = 0.5
    )
    + geom_point(
        data = area_shape_umap_df %>% dplyr::filter(Metadata_data_name == "mitocheck"),
        aes(color = Metadata_data_name),
        size = 0.5,
        alpha = 0.5
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
        "AreaShape\nfeatures only\n\nDataset",
        values = dataset_colors,
        labels = dataset_labels
    )
) 

area_shape_umap_gg

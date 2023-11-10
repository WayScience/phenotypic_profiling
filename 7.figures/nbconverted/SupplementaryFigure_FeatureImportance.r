suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(magick))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(ComplexHeatmap))

# Load variables important for plotting (e.g., themes, phenotypes, etc.)
source("themes.r")

# Set output directory
output_dir <- file.path("figures", "individual_coefficient_heatmaps")

heatmap_real_file <- file.path(output_dir, "compiled_real_coefficient_heatmaps.png")
heatmap_shuffled_file <- file.path(output_dir, "compiled_shuffled_coefficient_heatmaps.png")

# Load model coefficients
coef_dir <- file.path("../4.interpret_model/coefficients/")
coef_file <- file.path(coef_dir, "compiled_coefficients.tsv")

coef_df <- readr::read_tsv(
    coef_file,
    col_types = readr::cols(
        .default = "d",
        "Feature_Name" = "c",
        "Phenotypic_Class" = "c",
        "feature_type" = "c",
        "shuffled" = "c"
    )
) %>%
    dplyr::select(!`...1`) 

print(dim(coef_df))
head(coef_df, 3)

# Set constants for how to split feature name into interpretable parts
first_component_split <- c(
    "feature_space",
    "feature_name"
)

dp_feature_components <- c(
    "model",
    "component_idx"
)

cp_feature_components <- c(
    "feature_group",
    "measurement",
    "channel", 
    "parameter1", 
    "parameter2",
    "parameter3"
)

metadata_annotation_options <- list(
    "CP" = cp_feature_components,
    "DP" = dp_feature_components,
    "CP_and_DP" = dp_feature_components
)

# Create matrices for each combination of shuffled and feature type
feature_matrix_list <- list()
for (shuffled_value in unique(coef_df$shuffled)) {
    for (feature_type_value in unique(coef_df$feature_type)) {
        list_name_index <- paste0(feature_type_value, "__", shuffled_value)
        print(paste("Transforming model coefficients:", list_name_index))
        # Subset the coeficient dataframe
        coef_subset <- coef_df %>%
            dplyr::filter(
                shuffled == !!shuffled_value,
                feature_type == !!feature_type_value
            ) %>%
            dplyr::select(-c("shuffled", "feature_type")) %>%
            tidyr::pivot_wider(names_from = Feature_Name, values_from = Coefficent_Value)

        # Make data ready for matrix transformation
        coef_subset$Phenotypic_Class <- factor(
            coef_subset$Phenotypic_Class,
            levels=unique(coef_subset$Phenotypic_Class)
        )
        coef_subset_mat <- as.matrix(coef_subset[, -1])
        rownames(coef_subset_mat) <- coef_subset$Phenotypic_Class

        # Process metadata for inclusion in heatmap annotation
        metadata_subset <- dplyr::as_tibble(
            colnames(coef_subset_mat),
            .name_repair = function(x) "feature"
            ) %>%
            tidyr::separate(
                feature,
                into = first_component_split,
                sep = "__",
                remove = FALSE
            ) %>%
            tidyr::separate(
                "feature_name",
                into = metadata_annotation_options[[feature_type_value]],
                sep = "_",
                remove = FALSE
            )
        metadata_subset <- as.matrix(metadata_subset)
        rownames(metadata_subset) <- colnames(coef_subset_mat)
        
        # Store in list
        feature_matrix_list[[list_name_index]] <- list()
        feature_matrix_list[[list_name_index]][["coef_matrix"]] <- coef_subset_mat
        feature_matrix_list[[list_name_index]][["metadata_annotation"]] <- metadata_subset
    }
}

model_heatmap_file_names <- list()
for (model in names(feature_matrix_list)) {
    print(paste("Generating heatmap for:", model))

    # Create output file
    output_file <- file.path(output_dir, paste0("heatmap_", model, ".pdf"))

    # Store model file in a list for downstream loading with magick
    model_heatmap_file_names[model] <- output_file
    
    # Create components for plotheatmap_gg_list[["CP__False"]]ting and subsetting
    model_split_details <- unlist(stringr::str_split(model, "__"))
    feature_space <- model_split_details[1]
    shuffled_or_not <- model_split_details[2]

    if (shuffled_or_not == "False") {
        column_title = "Real data (final model)"
    } else {
        column_title = "Shuffled data"
    }

    # Generate heatmaps depending on the feature space
    if (feature_space == "CP") {
        column_title <- paste("CellProfiler features", column_title)
        coef_heatmap <- Heatmap(
            feature_matrix_list[[model]][["coef_matrix"]],
            top_annotation = HeatmapAnnotation(
                df = as.data.frame(feature_matrix_list[[model]][["metadata_annotation"]]) %>%
                    dplyr::select(feature_group),
                col = list(feature_group = cp_feature_group_colors),
                annotation_legend_param = list(feature_group = list(title = "CP feature\ngroup"))
            ),
            column_split = as.data.frame(feature_matrix_list[[model]][["metadata_annotation"]])$feature_group,
            column_title = column_title,
            name = "ML Coefficient",
            show_column_names = FALSE
        )

    } else if (feature_space == "CP_and_DP") {
        column_title <- paste("CP and DP features", column_title)
        coef_heatmap <- Heatmap(
            feature_matrix_list[[model]][["coef_matrix"]],
            top_annotation = HeatmapAnnotation(
                df = as.data.frame(feature_matrix_list[[model]][["metadata_annotation"]]) %>%
                    dplyr::select(feature_space),
                col = list(feature_space = feature_space_colors),
                annotation_legend_param = list(feature_space = list(title = "Feature\nspace"))
            ),
            column_title = column_title,
            name = "ML Coefficient",
            show_column_names = FALSE
        )
    } else {
        column_title <- paste("DeepProfiler features", column_title)
        coef_heatmap <- Heatmap(
            feature_matrix_list[[model]][["coef_matrix"]],
            top_annotation = HeatmapAnnotation(
                df = as.data.frame(feature_matrix_list[[model]][["metadata_annotation"]]) %>%
                    dplyr::select(feature_space),
                col = list(feature_space = feature_space_colors),
                annotation_legend_param = list(feature_space = list(title = "Feature\nspace"))
            ),
            column_split = as.data.frame(feature_matrix_list[[model]][["metadata_annotation"]])$feature_space,
            column_title = column_title,
            name = "ML Coefficient",
            show_column_names = FALSE
        )
    }

    pdf(output_file, width = 10, height = 6)
    draw(coef_heatmap, merge_legend = TRUE)
    dev.off()
}

model_heatmap_file_names

heatmap_gg_list <- list()
for (model in names(model_heatmap_file_names)) {
    model_file <- model_heatmap_file_names[[model]]
    heatmap_image <- magick::image_read(model_file)
    heatmap_gg_list[[model]] <- (
        ggplot()
        + annotation_custom(
            rasterGrob(image_trim(heatmap_image), interpolate = TRUE),
            xmin=-Inf,
            xmax=Inf,
            ymin=-Inf,
            ymax=Inf
        )
        + theme_void()
        + theme(
            plot.margin = margin(unit(c(0, 0, 0, 0), "lines"))
        )
    )
    print(paste("Converting to ggplot object:", model))
}

# Create heatmap patchwork
heatmap_top_gg <- (
    heatmap_gg_list[["CP__False"]] | heatmap_gg_list[["DP__False"]] | heatmap_gg_list[["CP_and_DP__False"]]
)
ggsave(heatmap_real_file, dpi = 600, height = 4.5, width = 12)

heatmap_bottom_gg <- (
    heatmap_gg_list[["CP__True"]] | heatmap_gg_list[["DP__True"]] | heatmap_gg_list[["CP_and_DP__True"]]
)
ggsave(heatmap_shuffled_file, dpi = 600, height = 4.5, width = 12)


heatmap_top_gg

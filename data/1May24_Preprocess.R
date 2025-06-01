# TPM Gene Expression Data Preprocessing
# Author: Hanchang (Hazel) Cai
# Date: May 24, 2025
library(tidyverse)
library(pheatmap)
library(RColorBrewer)
library(corrplot)

# 1. DATA LOADING =======
cancer_tpm_path <- "./data/pnas_tpm_96_nodup.txt"
cancer_tpm <- read.table(cancer_tpm_path, header = F, row.names = 1, sep = "\t", check.names = FALSE) #96 samples
normal_tpm_path <- "./data/pnas_normal_tpm.txt"
normal_tpm <- read.table(normal_tpm_path, header = T, row.names = 1, sep = "\t", check.names = FALSE) #32 samples
# Check if gene IDs match between datasets: ALL MATCHED
common_genes <- intersect(rownames(cancer_tpm), rownames(normal_tpm))
length(common_genes)
length(union(rownames(cancer_tpm), rownames(normal_tpm)))


# 2. DATA COMB.======
combined_tpm <- cbind(cancer_tpm, normal_tpm)

sample_info <- data.frame(
  sample_id = colnames(combined_tpm),
  condition = c(rep("Cancer", ncol(cancer_tpm)), rep("Normal", ncol(normal_tpm))),
  stringsAsFactors = FALSE
)
print(table(sample_info$condition))


# 3. DATA QC=======
sum(combined_tpm < 0, na.rm = TRUE) # NO negative TPM
sum(is.na(combined_tpm)) #NO missing data

# PLOT Distribution
hist(as.numeric(as.matrix(log(combined_tpm))))
hist(rowSums(combined_tpm >= 1)/ncol(combined_tpm))

# Remove genes with very low expression across all samples: 
# Thresh: TPM>1 in at least 10% of samples
gene_means <- rowMeans(combined_tpm, na.rm = TRUE)
low_expr_threshold <- 1 # TPM threshold
min_samples <- ceiling(0.1 * ncol(combined_tpm))
expressed_samples <- rowSums(combined_tpm >= 1) >= min_samples
keep_genes <- gene_means >= low_expr_threshold & expressed_samples

cat("# of genes before filtering:", nrow(combined_tpm), "\n") #60675
filtered_tpm <- combined_tpm[keep_genes, ]
cat("# of genes after filtering:", nrow(filtered_tpm), "\n") #50379

# 4. TPM DATA TRANSFORMATION=====
transform_log2_tpm <- function(tpm_matrix) {
  log2_tpm <- log2(tpm_matrix + 1)
  return(log2_tpm)
}
norm_log2_tpm <- transform_log2_tpm(filtered_tpm) # Log2(TPM + 1)

# 5. FEATURE SELECTION=======
# Select most variable genes: base on Var
select_variable_genes <- function(norm_matrix, n_genes = 5000) {
  gene_vars <- apply(norm_matrix, 1, var, na.rm = TRUE)
  # Remove genes with NA var
  gene_vars <- gene_vars[!is.na(gene_vars)]
  top_var_genes <- names(sort(gene_vars, decreasing = TRUE)[1:min(n_genes, length(gene_vars))])
  return(norm_matrix[top_var_genes, ])
}
ml_data_var <- select_variable_genes(norm_log2_tpm, n_genes = 5000)
dim(ml_data_var) #5000genes x 128samples

# 6. PCA ANALYSIS=======
perform_pca_analysis <- function(norm_data, method_name, sample_info) {
  # Remove any genes with 0 var
  gene_vars <- apply(norm_data, 1, var, na.rm = TRUE)
  non_zero_var_genes <- !is.na(gene_vars) & gene_vars > 0
  pca_data_clean <- norm_data[non_zero_var_genes, ]
  pca_input <- t(pca_data_clean)
  # Remove any columns with NA or infinite vals
  finite_cols <- apply(pca_input, 2, function(x) all(is.finite(x)))
  pca_input_clean <- pca_input[, finite_cols]
  # PCA
  pca_result <- prcomp(pca_input_clean, center = TRUE, scale. = TRUE)
  var_explained <- round(100 * pca_result$sdev^2 / sum(pca_result$sdev^2), 2)
  # PLOT
  pca_data <- data.frame(
    PC1 = pca_result$x[, 1],
    PC2 = pca_result$x[, 2],
    PC3 = pca_result$x[, 3],
    condition = sample_info$condition,
    sample_id = sample_info$sample_id
  )
  p1 <- ggplot(pca_data, aes(x = PC1, y = PC2, color = condition)) +
    geom_point(size = 3, alpha = 0.8) +
    stat_ellipse(level = 0.68, linetype = "dashed") +  # Add confidence ellipses
    labs(
      title = paste("PCA Analysis -", method_name),
      x = paste0("PC1 (", var_explained[1], "% variance)"),
      y = paste0("PC2 (", var_explained[2], "% variance)"),
      color = "Condition"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = "bottom"
    ) +
    scale_color_manual(values = c("Cancer" = "#E74C3C", "Normal" = "#3498DB"))
  scree_data <- data.frame(
    PC = 1:min(20, length(var_explained)),
    Variance = var_explained[1:min(20, length(var_explained))]
  )
  
  p2 <- ggplot(scree_data, aes(x = PC, y = Variance)) +
    geom_line(color = "#2C3E50", size = 1) +
    geom_point(color = "#E74C3C", size = 2) +
    labs(
      title = paste("Scree Plot -", method_name),
      x = "Principal Component",
      y = "Variance Explained (%)"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  print(p1)
  print(p2)
  cat("Variance explained by first 10 PCs:\n")
  print(var_explained[1:min(10, length(var_explained))])
  return(list(
    pca_result = pca_result,
    pca_data = pca_data,
    var_explained = var_explained
  ))
}

pca_log2_var <- perform_pca_analysis(ml_data_var, "Log2(TPM+1) - Most Variable", sample_info)

# 7. SAMPLE CORRELATION ANALYSIS=======
sample_cors <- cor(ml_data_var, method = "pearson")
pheatmap(
  sample_cors,
  annotation_col = data.frame(
    Condition = sample_info$condition,
    row.names = sample_info$sample_id
  ),
  annotation_colors = list(
    Condition = c("Cancer" = "#E74C3C", "Normal" = "#3498DB")
  ),
  main = "Sample-to-Sample Correlation Heatmap",
  show_rownames = FALSE,
  show_colnames = FALSE
)


# 8. PREPARE DATASETS=====
prepare_ml_dataset <- function(norm_data, sample_info, method_name) {
  # Transpose: samples = rows, genes = columns
  ml_features <- t(norm_data)
  ml_features[is.na(ml_features)] <- 0
  # TARGET: 0 = Normal, 1 = Cancer
  ml_target <- ifelse(sample_info$condition == "Cancer", 1, 0)
  ml_dataset <- data.frame(
    ml_features,
    target = ml_target,
    condition = sample_info$condition,
    check.names = FALSE
  )
  
  cat("ML Dataset prepared for", method_name, ":\n")
  cat("Dimensions:", dim(ml_dataset), "\n")
  cat("Features:", ncol(ml_dataset) - 2, "\n")  # Exclude target and condition columns
  cat("Target distribution:\n")
  print(table(ml_dataset$target))
  
  return(ml_dataset)
}
ml_dataset_var <- prepare_ml_dataset(ml_data_var, sample_info, "Most Variable Genes")

save(
  combined_tpm,
  filtered_tpm,
  norm_log2_tpm,
  ml_data_var,
  ml_dataset_var,
  sample_info,
  pca_log2_var,
  file = "processed_tpm_data.RData"
)
# MAIN dataset
write.csv(ml_dataset_var, "combined_tpm_5000genes.csv", row.names = FALSE)
# sample_info
write.csv(sample_info, "sample_info_tpm.csv", row.names = FALSE) 
# gene lists
write.csv(
  data.frame(gene_id = rownames(ml_data_var)),
  "selected_variable_genes.csv",
  row.names = FALSE
)

# ==============================================================================
# Validation dataset
# ==============================================================================
library(readxl)
validation_tpm_path <- "./data/validation_exon_tpm"
validation_tpm <- read.table(validation_tpm_path, header = TRUE, row.names = 1, sep = "\t", check.names = FALSE)
# Load metadata files
bc_meta_path <- "./data/validation_bc_meta.xlsx"
normal_meta_path <- "./data/validation_normal_meta.xlsx"
bc_meta <- read_excel(bc_meta_path)
normal_meta <- read_excel(normal_meta_path)
bc_sample_ids <- bc_meta$`Mapping ID`
normal_sample_ids <- normal_meta$`Mapping ID`
# Find matches
tpm_sample_ids <- colnames(validation_tpm)
bc_matches <- intersect(tpm_sample_ids, bc_sample_ids)
normal_matches <- intersect(tpm_sample_ids, normal_sample_ids)
setdiff(tpm_sample_ids, c(bc_matches, normal_matches))
validation_sample_info <- data.frame(
  sample_id = c(bc_matches, normal_matches),
  condition = c(rep("Cancer", length(bc_matches)), rep("Normal", length(normal_matches))),
  dataset = "Validation",
  stringsAsFactors = FALSE
)
table(validation_sample_info$condition)

matched_samples <- c(bc_matches, normal_matches)
validation_tpm_matched <- validation_tpm[, matched_samples]
dim(validation_tpm_matched) #60675   161
validation_genes <- rownames(validation_tpm_matched)
common_genes_val <- intersect(training_genes, validation_genes)
validation_tpm_aligned <- validation_tpm_matched[common_genes_val, ]
# Preprocess validation data: same pipeline as training
validation_log2_tpm <- transform_log2_tpm(validation_tpm_aligned)

# Prepare datasets using different normalization methods
validation_ml_dataset <- prepare_ml_dataset(
  validation_log2_tpm, 
  validation_sample_info, 
  "Validation - Log2(TPM+1)"
)

validation_pca <- perform_pca_analysis(
  validation_log2_tpm, 
  "Validation - Log2(TPM+1)", 
  validation_sample_info
)
# COMPARE WITH TRAINING DATA DISTRIBUTION

if(exists("norm_log2_tpm")) {
  training_subset <- norm_log2_tpm[common_genes_val, ]
  training_means <- rowMeans(training_subset, na.rm = TRUE)
  validation_means <- rowMeans(validation_log2_tpm, na.rm = TRUE)
  # Correlation between mean expressions
  mean_correlation <- cor(training_means, validation_means, use = "complete.obs")
  # Plot
  plot(training_means, validation_means, 
       xlab = "Training Mean Expression", 
       ylab = "Validation Mean Expression",
       main = paste("Gene Expression Correlation\nr =", round(mean_correlation, 3)),
       pch = 16, alpha = 0.5)
  abline(0, 1, col = "red", lty = 2)
}

# Save CSV files for Validation dataset
write.csv(validation_ml_dataset, "validation_ml_dataset_log2.csv", row.names = FALSE)

write.csv(validation_sample_info, "validation_sample_info.csv", row.names = FALSE)

sample_cors <- cor(validation_log2_tpm, method = "pearson")

# Create correlation heatmap
pheatmap(
  sample_cors,
  annotation_col = data.frame(
    Condition = validation_sample_info$condition,
    row.names = validation_sample_info$sample_id
  ),
  annotation_colors = list(
    Condition = c("Cancer" = "#E74C3C", "Normal" = "#3498DB")
  ),
  main = "Sample-to-Sample Correlation Heatmap",
  show_rownames = FALSE,
  show_colnames = FALSE
)
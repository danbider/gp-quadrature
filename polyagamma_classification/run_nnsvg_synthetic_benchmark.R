args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript run_nnsvg_synthetic_benchmark.R <input_dir> <output_csv> [install_missing=true|false]")
}

input_dir <- args[[1]]
output_csv <- args[[2]]
install_missing <- if (length(args) >= 3) {
  tolower(args[[3]]) %in% c("true", "1", "yes")
} else {
  FALSE
}

ensure_bioc_package <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(invisible(TRUE))
  }
  if (!install_missing) {
    stop(sprintf("Required package '%s' is not installed.", pkg))
  }
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  BiocManager::install(pkg, ask = FALSE, update = FALSE)
}

ensure_bioc_package("SpatialExperiment")
ensure_bioc_package("nnSVG")
ensure_bioc_package("scuttle")

suppressPackageStartupMessages({
  library(SpatialExperiment)
  library(nnSVG)
  library(scuttle)
  library(SummarizedExperiment)
})

counts_path <- file.path(input_dir, "sim_counts.csv")
spots_path <- file.path(input_dir, "spots.csv")

counts_df <- read.csv(counts_path, row.names = 1, check.names = FALSE)
spots_df <- read.csv(spots_path, check.names = FALSE)

counts_mat <- as.matrix(counts_df)
storage.mode(counts_mat) <- "numeric"

if (!all(colnames(counts_mat) == spots_df$spot_id)) {
  stop("Counts columns must match spot_id order in spots.csv")
}

zero_gene <- rowSums(counts_mat) == 0
if (any(zero_gene)) {
  counts_mat <- counts_mat[!zero_gene, , drop = FALSE]
}

zero_spot <- colSums(counts_mat) == 0
if (any(zero_spot)) {
  counts_mat <- counts_mat[, !zero_spot, drop = FALSE]
  spots_df <- spots_df[!zero_spot, , drop = FALSE]
}

spatial_coords <- as.matrix(spots_df[, c("x", "y")])

spe <- SpatialExperiment(
  assays = list(counts = counts_mat),
  spatialCoords = spatial_coords
)

rownames(spe) <- rownames(counts_mat)
colnames(spe) <- colnames(counts_mat)

spe <- computeLibraryFactors(spe)
spe <- logNormCounts(spe)

set.seed(1)
t0 <- proc.time()[["elapsed"]]
spe <- nnSVG(spe, assay_name = "logcounts")
runtime_sec <- proc.time()[["elapsed"]] - t0

res <- as.data.frame(rowData(spe))
res$gene_id <- rownames(spe)
res$nnsvg_runtime_total_sec <- runtime_sec
res$nnsvg_runtime_per_gene_sec <- runtime_sec / nrow(res)

write.csv(res, output_csv, row.names = FALSE)
cat(sprintf("Wrote %s\n", output_csv))

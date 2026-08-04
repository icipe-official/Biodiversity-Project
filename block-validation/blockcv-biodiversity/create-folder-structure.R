# Target absolute path
project_dir <- "/home/vincent/Development/geospatial-analysis/Biodiversity-Project/blockcv-biodiversity"

# 1. Create directory structure
dirs <- c(
  file.path(project_dir, "R"),
  file.path(project_dir, "src"),
  file.path(project_dir, "data", "input"),
  file.path(project_dir, "data", "output")
)

sapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE)

# 2. Helper function to write files
create_file <- function(path, content = "") {
  writeLines(content, con = file.path(project_dir, path))
}

# 3. Populate root files
create_file(".gitignore", ".Rproj.user\n.Rhistory\n.RData\n*.rds\ndata/input/*\n!data/input/.gitkeep\ndata/output/*\n!data/output/.gitkeep")
create_file("data/input/.gitkeep", "")
create_file("data/output/.gitkeep", "")

create_file("README.md", "# blockcv-biodiversity\n\nSpatial Block Cross-Validation Pipeline for Biodiversity Modeling.")

create_file("Dockerfile", "FROM rocker/geospatial:4.3.0\nWORKDIR /app\nCOPY . /app\nCMD [\"Rscript\", \"main.R\"]")

create_file("config.R", "CONFIG <- list(\n  seed = 42,\n  k_folds = 5,\n  paths = list(input = 'data/input', output = 'data/output')\n)")

create_file("main.R", "source('config.R')\nsource('R/utils.R')\n\nmessage('Running pipeline...')\n\nlapply(list.files('src', full.names = TRUE), source)\nmessage('Done!')")

# 4. Populate helper and src files
create_file("R/utils.R", "# Utility functions\nload_spatial <- function(path) { sf::st_read(path) }")

src_files <- c(
  "00_preprocess_occurrences.R",
  "01_data_prep.R",
  "02_block_cv.R",
  "03_model_training.R",
  "04_predict_ensemble.R",
  "05_compute_ibii_exports.R"
)

for (f in src_files) {
  create_file(file.path("src", f), sprintf("# Step: %s\nmessage('Executing %s')", f, f))
}

cat("Project created successfully at:", normalizePath(project_dir), "\n")
# =============================================================================
# One-time setup: installs the R packages used by the dashboard into a
# project-local library (dashboard/r-lib) so nothing is installed globally.
#
# Usage:  Rscript setup.R
# =============================================================================

script_dir <- function() {
  cmd <- commandArgs()
  if (any(nzchar(cmd) & grepl("--file=", cmd, fixed = TRUE))) {
    f <- sub("^--file=", "", cmd[grepl("--file=", cmd, fixed = TRUE)][1])
    return(normalizePath(dirname(f), mustWork = FALSE))
  }
  getwd()
}

dir <- script_dir()
lib <- file.path(dir, "r-lib")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))

options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 600)

needed <- c(
  "shiny", "shinydashboard", "httr", "jsonlite",
  "DT", "plotly", "dplyr", "tidyr", "ggplot2"
)
installed <- rownames(installed.packages())
missing <- setdiff(needed, installed)

if (length(missing) > 0) {
  message("Installing into ", lib, ": ", paste(missing, collapse = ", "))
  install.packages(missing)
} else {
  message("All packages already installed in ", lib)
}
message("Done. Launch the dashboard with:  Rscript run.R")

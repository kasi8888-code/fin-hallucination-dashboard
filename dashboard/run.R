# =============================================================================
# Launcher for the Hallucination Risk Shiny dashboard.
#
# Bootstraps the project-local R package library (dashboard/r-lib) so no
# global R package install is required, then runs the app.
#
# Usage:
#   Rscript run.R
#   Rscript run.R 8000            # launch on a specific port
# =============================================================================

# Directory of this script (works whether run as `Rscript run.R`, sourced,
# or opened in RStudio).
get_script_dir <- function() {
  cmd <- commandArgs()
  if (any(nzchar(cmd) & grepl("--file=", cmd, fixed = TRUE))) {
    # Rscript path
    f <- sub("^--file=", "", cmd[grepl("--file=", cmd, fixed = TRUE)][1])
    return(normalizePath(dirname(f), mustWork = FALSE))
  }
  # RStudio / source() path
  frame_files <- Filter(function(x) !is.null(x),
                        lapply(sys.frames(), function(f) f$ofile))
  if (length(frame_files) > 0) {
    return(normalizePath(dirname(frame_files[[1]]), mustWork = FALSE))
  }
  getwd()
}

args <- commandArgs(trailingOnly = TRUE)
port <- if (length(args) >= 1) as.integer(args[1]) else 4200

script_dir <- get_script_dir()
lib <- file.path(script_dir, "r-lib")
if (dir.exists(lib)) .libPaths(c(lib, .libPaths()))

message("Starting Hallucination Risk dashboard on http://127.0.0.1:", port)
shiny::runApp(script_dir, host = "127.0.0.1", port = port,
              launch.browser = TRUE)

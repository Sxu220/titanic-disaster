# Installs the R packages used by titanic.R.
# The rocker/r-ver base image points at a dated Posit Package Manager snapshot,
# so these installs resolve to fixed versions every time the image is built.

packages <- c("readr", "dplyr")

install.packages(packages, repos = getOption("repos"))

for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop(paste("Package failed to install:", p))
  }
  cat(sprintf("Installed %s %s\n", p, as.character(packageVersion(p))))
}

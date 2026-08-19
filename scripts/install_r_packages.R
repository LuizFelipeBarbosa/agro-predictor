options(repos = c(CRAN = "https://cloud.r-project.org"))

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}

install_if_missing("sits")
install_if_missing("arrow")    # required by the pysits bridge
install_if_missing("remotes")

# Training samples live in the GitHub-only sitsdata package (large download).
if (!requireNamespace("sitsdata", quietly = TRUE)) {
  remotes::install_github("e-sensing/sitsdata")
}

for (pkg in c("sits", "arrow", "sitsdata")) {
  cat(pkg, as.character(packageVersion(pkg)), "\n")
}

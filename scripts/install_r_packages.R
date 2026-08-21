options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 600)

# Downloads can fail transiently; retry each package up to 3 times.
# install.packages() re-attempts any still-missing dependencies on retry.
install_with_retry <- function(pkg, installer) {
  for (attempt in 1:3) {
    if (requireNamespace(pkg, quietly = TRUE)) return(invisible())
    cat("==> installing", pkg, "(attempt", attempt, ")\n")
    try(installer())
  }
  if (!requireNamespace(pkg, quietly = TRUE)) stop("failed to install ", pkg)
}

install_with_retry("sits", function() install.packages("sits"))
install_with_retry("arrow", function() install.packages("arrow"))  # pysits bridge
install_with_retry("remotes", function() install.packages("remotes"))

# sits lists these as Suggests, but pysits needs kohonen at import time and
# sits_rfor() needs randomForest at training time.
install_with_retry("kohonen", function() install.packages("kohonen"))
install_with_retry("randomForest", function() install.packages("randomForest"))
install_with_retry("e1071", function() install.packages("e1071"))
install_with_retry("caret", function() install.packages("caret"))  # sits_kfold_validate

# Training samples live in the GitHub-only sitsdata package (large download).
install_with_retry("sitsdata", function() remotes::install_github("e-sensing/sitsdata"))

# sits loads torch at classification time even for non-DL models; torch needs
# its native libtorch/lantern binaries downloaded once after install.
if (!torch::torch_is_installed()) torch::install_torch()

for (pkg in c("sits", "arrow", "sitsdata")) {
  cat(pkg, as.character(packageVersion(pkg)), "\n")
}

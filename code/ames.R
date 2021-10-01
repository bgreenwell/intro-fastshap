library(fastshap)
library(ggplot2)
library(isotree)
library(ranger)


# Read in Ames housing data and split into train/test sets using a 70/30 split
ames <- as.data.frame(AmesHousing::make_ames())
ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
set.seed(4919)  # for reproducibility
id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[id, ]  # 2051 x 81
ames.tst <- ames[-id, ]  # 879 x 81

# Fit a (default) random forest
set.seed(945)  # for reproducibility
ames.rfo <- ranger(Sale_Price ~ ., data = ames.trn, 
                   importance = "permutation")



# Prediction wrapper that tells fastshap how to obtain predictions for new
# observations
pfun <- function(object, newdata) {
  predict(object, data = newdata)$predictions
}

# Data frame of just feature columns
xnames <- setdiff(names(ames.trn), "Sale_Price")
X <- ames.trn[, xnames]  # feature columns only

# Generate feature contributions
system.time({
  set.seed(1351)  # for reproducibility
  (ex <- explain(ames.rfo, X = X, pred_wrapper = pfun, 
                 adjust = TRUE, nsim = 100, .progress = "text"))
})

# Save results
saveRDS(ex, file = "data/ranger-shap-ames-tst.rds")

p1 <- autoplot(ex)
p2 <- autoplot(ex, type = "dependence", feature = "Gr_Liv_Area", X = X, 
               alpha = 0.5)
p3 <- autoplot(ex, type = "contribution", row_num = 1, X = ames.tst, 
               feature_values = ames.tst[1, xnames])

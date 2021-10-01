library(ranger)

# Read in the data and clean it up a bit
titanic <- titanic::titanic_train
features <- c(
  "Survived",  # passenger survival indicator
  "Pclass",    # passenger class
  "Sex",       # gender
  "Age",       # age
  "SibSp",     # number of siblings/spouses aboard
  "Parch",     # number of parents/children aboard
  "Fare",      # passenger fare
  "Embarked"   # port of embarkation
)
titanic <- titanic[, features]
titanic$Survived <- as.factor(titanic$Survived)
titanic <- na.omit(titanic)

# Data frame containing just the features
X <- subset(titanic, select = -Survived)

fit <- ranger(Survived ~ ., data = titanic, probability = TRUE)

jack <- data.frame(
  Pclass = 3,
  Sex = factor("male", levels = c("female", "male")),
  Age = 20,
  SibSp = 0,
  Parch = 0,
  Fare = 15,  # lower end of third-class ticket prices
  Embarked = factor("S", levels = c("", "C", "Q", "S"))
)

predict(fit, data = jack)$predictions

# Prediction wrapper to compute predcited probability of survive
pfun <- function(object, newdata) {
  predict(object, data = newdata)$predictions[, "1"]
}

# DALEX-based helper for iBreakDown
explainer <- DALEX::explain(fit, data = X, y = titanic$Survived, 
                            predict_function = pfun, verbose = FALSE)

# Helper for iml
predictor <- iml::Predictor$new(fit, data = titanic, y = "Survived",
                                predict.fun = pfun)

# Compute explanations
set.seed(1039)  # for reproducibility
ex1 <- iBreakDown::shap(explainer, B = 100, new_observation = jack)
ex2 <- iml::Shapley$new(predictor, x.interest = jack, sample.size = 100)
ex3 <- fastshap::explain(fit, X = X, pred_wrapper = pfun, nsim = 100,
                         newdata = jack)

# Plot results
library(ggplot2)  # for `autoplot()` function
p3 <- plot(ex1) + ggtitle("iBreakDown")
p2 <- plot(ex2) + ggtitle("iml")
p1 <- autoplot(ex3, type = "contribution") + ggtitle("fastshap")
fastshap::grid.arrange(p1, p2, p3, nrow = 1)

nsims <- c(1, 5, 10, 25, 50, 75, 100, seq(from = 110, to = 1000, by = 10))
times1 <- times2 <- times3 <- numeric(length(nsims))
set.seed(904)
for (i in seq_along(nsims)) {
  message("nsim = ", nsims[i], "...")
  times1[i] <- system.time({
    iBreakDown::shap(explainer, B = nsims[i], new_observation = jack, keep_distribution=F)
  })["elapsed"]
  times2[i] <- system.time({
    iml::Shapley$new(predictor, x.interest = jack, sample.size = nsims[i])
  })["elapsed"]
  times3[i] <- system.time({
    fastshap::explain(fit, X = X, newdata = jack, pred_wrapper = pfun, 
                      nsim = nsims[i])
  })["elapsed"]
}

# Save results
res <- data.frame(
  "nsim" = nsims,
  "iBreakDown" = times1,
  "iml" = times2,
  "fastshap" = times3
)
saveRDS(res, file = "data/benchmark-results.rds")

# Plot results
palette("Okabe-Ito")
plot(res$nsim, res$iBreakDown, type = "b", xlab = "Number of Monte Carlo repetitions",
     ylab = "Time (in seconds)", las = 1, col = 2,
     xlim = c(0, max(nsims)), 
     ylim = c(min(res), max(times1, times2, times3)), log = "y")
     # ylim = c(0, max(times1, times2, times3)))
     # ylim = c(0, 10))
lines(res$nsim, res$iml, type = "b", col = 3)
lines(res$nsim, res$fastshap, type = "b", col = 4)
legend("right", bty = "n",
       legend = c("iBreakDown", "iml", "fastshap"),
       lty = 1, pch = 1, col = c(2, 3, 4), inset = 0.02)
abline(h = 0, lty = 2)
palette("default")

rbind(
  "iBreakDown" = coef(lm(iBreakDown ~ nsim + 1, data = res)),
  "iml" = coef(lm(iml ~ nsim + 1, data = res)),
  "fastshap" = coef(lm(fastshap ~ nsim + 1, data = res))
)

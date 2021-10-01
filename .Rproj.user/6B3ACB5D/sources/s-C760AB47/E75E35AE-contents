library(fastshap)
library(ggplot2)
library(ranger)
library(waterfall)

# Set ggplot2 theme
theme_set(theme_bw())

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

# Fit a (default) random forest
set.seed(1046)  # for reproducibility
rfo <- ranger(Survived ~ ., data = titanic, probability = TRUE)
ladd <- function(x, data = NULL, ..., plot = trellis.last.object()) {
  return(plot + eval(substitute(latticeExtra::layer(foo, data = data, ...), 
                                list(foo = substitute(x)))))
}
# Prediction wrapper for `fastshap::explain()`; has to return a single
# (atomic) vector of predictions
pfun <- function(object, newdata) {  # compute prob(survived=1|x)
  predict(object, data = newdata)$predictions[, 2]
}

# Estimate feature contributions for each imputed training set
X <- subset(titanic, select = -Survived)  # features only!
set.seed(1051)
(ex.all <- 
   explain(rfo, X = X, nsim = 100, adjust = TRUE, 
           pred_wrapper = pfun, .progress = "text"))

# Construct some plots
p1 <- autoplot(ex.all)
p2 <- autoplot(ex.all, type = "dependence", feature = "Age", X = X,
               color_by = "Sex") + theme(legend.position = c(0.8, 0.8))
gridExtra::grid.arrange(p1, p2, nrow = 1)

# Explain an individual passenger
jack.dawson <- data.frame(
  # Survived = factor(0, levels = 0:1),  # in case you haven't seen the movie
  Pclass = 3,
  Sex = factor("male", levels = c("female", "male")),
  Age = 20,
  SibSp = 0,
  Parch = 0,
  Fare = 15,  # lower end of third-class ticket prices
  Embarked = factor("S", levels = c("", "C", "Q", "S"))
)

# Estimate feature contributions for Jack's predicted probability
set.seed(754)
(ex.jack <- 
    explain(rfo, X = X, newdata = jack.dawson, nsim = 1000, 
          adjust = TRUE, pred_wrapper = pfun))


# Waterfall chart of feature contributions
res <- data.frame(
  "feature" = paste0(names(jack.dawson), "=", t(jack.dawson)),
  "shapley.value" = t(ex.jack)
)
pred.jack <- pfun(rfo, newdata = jack.dawson)
baseline <- mean(pfun(rfo, newdata = X))  # avg training prediction
palette("Okabe-Ito")
waterfallchart(feature ~ shapley.value, data = res, origin = baseline,
               summaryname = "f(x) - baseline", col = 2:3,
               xlab = "Probability of survival")
ladd(panel.abline(v = pred.jack, lty = 2, col = 1))
ladd(panel.abline(v = baseline, lty = 2, col = 1))
ladd(panel.text(0.10, 4, labels = "f(x)", col = 1))
ladd(panel.text(0.385, 4, labels = "baseline", col = 1))
palette("default")

library(fastshap)
library(ggplot2)
library(isotree)


# Set ggplot2 theme
theme_set(theme_bw())

# Read in credit card fraud data
ccfraud <- data.table::fread("data/ccfraud.csv")

# Randomize the data
set.seed(2117)  # for reproducibility
ccfraud <- ccfraud[sample(nrow(ccfraud)), ]

# Split data into train/test sets
set.seed(2013)  # for reproducibility
trn.id <- sample(nrow(ccfraud), size = 10000, replace = FALSE)
ccfraud.trn <- ccfraud[trn.id, ]
ccfraud.tst <- ccfraud[-trn.id, ]

# Check class distribution in each
proportions(table(ccfraud.trn$Class))
proportions(table(ccfraud.tst$Class))

# Fit a default isolation forest
ifo <- isolation.forest(ccfraud.trn[, 1L:30L], random_seed = 2223, nthreads = 1)

# Compute anomaly scores for the test observations
head(scores <- predict(ifo, newdata = ccfraud.tst))

# Store results
saveRDS(ifo, file = "data/rf-ccfraud-ifo.rds")
saveRDS(scores, file = "data/rf-ccfraud-scores-test.rds")

# pred <- prediction(scores, ccfraud.tst$Class)
# gain <- performance(pred, "tpr", "rpp")
# plot(gain, col = "orange")

# Plot cumulative gain chart
ord <- order(scores, decreasing = TRUE)
y <- ccfraud.tst$Class[ord]  # order according to sorted scores
prop <- seq_along(y) / length(y)
lift <- cumsum(y) / sum(ccfraud.tst$Class)  # convert to proportion
ccfraud.lift <- data.frame(prop, lift)
palette("Okabe-Ito")
ggplot(ccfraud.lift, aes(prop, lift)) +
  geom_line(color = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, color = 1) +
  scale_x_continuous(breaks = 0:10 / 10) +
  scale_y_continuous(breaks = 0:10 / 10) +
  xlab("Proportion of sample inspected") +
  ylab("Proportion of anomalies identified")
palette("default")

# Compute training scores and average to get baseline
scores.trn <- predict(ifo, newdata = ccfraud.trn)
to.explain <- max(scores) - mean(scores.trn)  # the difference we're explaining

# Find test observations corresponding to maximum anomaly score
max.id <- which.max(scores)  # row ID for observation wit
max.x <- ccfraud.tst[max.id, ]
max(scores)
max.x

X <- ccfraud.trn[, 1L:30L]  # feature columns only!
max.x <- max.x[, 1L:30L]  # feature columns only!
pfun <- function(object, newdata) {  # prediction wrapper
  predict(object, newdata = newdata)
}

# Generate feature contributions
set.seed(1351)  # for reproducibility
(ex <- explain(ifo, X = X, newdata = max.x, pred_wrapper = pfun, 
               adjust = TRUE, nsim = 1000))
sum(ex)  # should sum to f(x) - baseline whenever `adjust = TRUE` 

res <- data.frame(
  "feature" = names(ex),
  "shapley.value" = as.numeric(as.vector(ex[1L,]))
)

ggplot(res, aes(x = shapley.value, y = reorder(feature, shapley.value))) +
  geom_point() +
  geom_vline(xintercept = 0, linetype = "dashed") +
  xlab("Shapley value") +
  ylab("") +
  theme(axis.text.y = element_text(size = rel(0.8)))

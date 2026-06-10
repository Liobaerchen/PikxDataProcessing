install.packages("robustbase")
library(robustbase)

data <- read.csv("threepluslivwell.csv")

mm_model <- lmrob(
  dv ~ heat + rain + factor(region_id) + factor(year),
  data = data
)

summary(mm_model)

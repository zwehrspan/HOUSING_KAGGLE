library(tidyverse)
library(tidymodels)

set.seed(123)
fullTrainData <- read.csv("../house-prices-advanced-regression-techniques/train.csv")
finalTestData <- read.csv("../house-prices-advanced-regression-techniques/test.csv")
finalTestDataORI <- read.csv("../house-prices-advanced-regression-techniques/test.csv")

theme_set(theme_bw())


# Recipes for imputing values, normalize
dataRecipe <- recipe(fullTrainData, SalePrice ~ .) %>%
    step_rm(Id, Street, Utilities) %>% 
    step_log(all_numeric(),-all_outcomes(), offset = 1) %>%
    step_normalize(all_numeric(),-all_outcomes()) %>%
    step_other(all_nominal(), -all_outcomes(), threshold = 0.01) %>%
    step_novel(all_predictors(), -all_numeric()) %>%
    step_impute_knn(all_predictors()) %>%
    step_dummy(all_nominal(), -all_outcomes()) 



# Predict Sales price

# Split into training and testing for the model
dataSplit <-
    initial_split(fullTrainData, prop = 3 / 4,
                  strata = SalePrice)

trainData <- training(dataSplit)
testData  <- testing(dataSplit)

dia_vfold <- vfold_cv(trainData, v = 10, repeats = 5, strata = SalePrice)

# Try a ranger model

rfModelRanger <-
    rand_forest(
        mode = "regression",
        engine = "ranger",
        mtry = tune(),
        trees = tune(),
        min_n = tune()
    )


dataWF <-
    workflow() %>%
    add_model(rfModelRanger) %>%
    add_recipe(dataRecipe)

salesTune <-
    dataWF %>%
    tune_grid(
        dia_vfold,
        grid = grid_max_entropy(
            mtry(range = c(15, 25)),
            trees(range = c(1000,10000)),
            min_n(range = c(1,3)),
            size = 5) ,
        metrics = metric_set(rmse)
    )
salesTune %>% show_best()
autoplot(salesTune)

SalePriceBestModel <- select_best(salesTune, "rmse", maximize = F)
SalePriceFinalModel <- finalize_model(rfModelRanger, SalePriceBestModel)
SalePriceWorkflow    <- dataWF %>% update_model(SalePriceFinalModel)
SalePriceFinalFit     <- fit(SalePriceWorkflow, data = trainData)

rfTrainingPred <-
    predict(SalePriceFinalFit, trainData) %>%
    bind_cols(trainData %>%
                  select(SalePrice))
colnames(rfTrainingPred) <- c("estimate", "truth")
ggplot(rfTrainingPred, aes(x = truth, estimate)) +
    geom_point(alpha = .2) +
    xlab("Known Sale Price (USD)") +
    ylab("Predicted Sale Price (USD)") +
    geom_abline(slope = 1, intercept = 0) +
    theme_bw()

rfTestingPred <-
    predict(SalePriceFinalFit, finalTestData)

finalOutput <- cbind.data.frame(finalTestDataORI$Id, 
                                rfTestingPred$.pred)
colnames(finalOutput) <- c("Id", "SalePrice")
write_csv(finalOutput, "submission.csv")


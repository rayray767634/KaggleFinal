library(vroom)
library(tidyverse)
library(dplyr)
library(patchwork)
library(tidymodels)
library(glmnet)
library(stacks)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)
library(bonsai)
library(lightgbm)
library(dbarts)

ottotest <- vroom("finaltest.csv")

ottotrain <- vroom("finaltrain.csv")


my_recipe <- recipe(target~.,data = ottotrain) %>%
  step_rm(id) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor)
  #step_lencode_glm(all_nominal_predictors(), outcome = vars(target)) %>%
  #step_normalize(all_predictors()) %>%
  #step_pca(all_predictors(), threshold = .9)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = ottotrain)
View(baked)

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
# Set up grid of tuning values
tuning_grid_nb <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

# Set up K-fold CV
folds <- vfold_cv(ottotrain, v = 5, repeats = 1)
# Run the CV
CV_results_nb <-nb_wf  %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_nb,
            metrics = metric_set(accuracy))

# Find best tuning parameters
bestTune_nb <- CV_results_nb %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf_nb <- 
  nb_wf %>%
  finalize_workflow(bestTune_nb) %>%
  fit(data = ottotrain)

# predict
otto_nb_predictions <- predict(final_wf_nb,
                              new_data = ottotest,
                              type = "prob") %>%
  bind_cols(., ottotest) %>% #Bind predictions with test data
  select(id, .pred_Class_1,.pred_Class_2,.pred_Class_3,.pred_Class_4,.pred_Class_5,
         .pred_Class_6,.pred_Class_7,.pred_Class_8,.pred_Class_9,) %>% 
  rename(Class_1=.pred_Class_1,Class_2=.pred_Class_2,Class_3=.pred_Class_3,
         Class_4=.pred_Class_4,Class_5=.pred_Class_5,Class_6=.pred_Class_6,Class_7=.pred_Class_7,
         Class_8=.pred_Class_8,Class_9=.pred_Class_9,)

vroom_write(x=otto_nb_predictions, file="./OTTONBPreds.csv", delim=",")




## boost

rec <- recipe(target ~ ., data = ottotrain) %>%
  update_role(id, new_role = "Id") %>%
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE)

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>% # or "xgboost" but lightgbm is faster
  set_mode("classification")

bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>%
  set_mode("classification")


boost_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(boost_model)

boost_tuneGrid <- grid_regular(tree_depth(),
                               trees(),
                               learn_rate(),
                               levels = 5)
folds <- vfold_cv(ottotrain, v = 5, repeats = 1)


CV_results_boost <-boost_wf  %>%
  tune_grid(resamples = folds,
            grid = boost_tuneGrid,
            metrics = metric_set(accuracy))

# Find best tuning parameters
bestTune_boost <- CV_results_boost %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf_boost <- 
  boost_wf %>%
  finalize_workflow(bestTune_boost) %>%
  fit(data = ottotrain)

# predict
otto_boost_predictions <- predict(final_wf_boost,
                               new_data = ottotest,
                               type = "prob") %>%
  bind_cols(., ottotest) %>% #Bind predictions with test data
  select(id, .pred_Class_1,.pred_Class_2,.pred_Class_3,.pred_Class_4,.pred_Class_5,
         .pred_Class_6,.pred_Class_7,.pred_Class_8,.pred_Class_9,) %>% 
  rename(Class_1=.pred_Class_1,Class_2=.pred_Class_2,Class_3=.pred_Class_3,
         Class_4=.pred_Class_4,Class_5=.pred_Class_5,Class_6=.pred_Class_6,Class_7=.pred_Class_7,
         Class_8=.pred_Class_8,Class_9=.pred_Class_9,)

vroom_write(x=otto_boost_predictions, file="./OTTOBOOSTPreds.csv", delim=",")

## bart

rec <- recipe(target ~ ., data = ottotrain) %>%
  update_role(id, new_role = "Id") %>%
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE)

bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>%
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(bart_model)

bart_tuneGrid <- grid_regular(trees(),
                               levels = 5)

folds <- vfold_cv(ottotrain, v = 5, repeats = 1)


CV_results_bart <-bart_wf  %>%
  tune_grid(resamples = folds,
            grid = bart_tuneGrid,
            metrics = metric_set(accuracy))

# Find best tuning parameters
bestTune_bart <- CV_results_bart %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf_bart <- 
  bart_wf %>%
  finalize_workflow(bestTune_bart) %>%
  fit(data = ottotrain)

# predict
otto_bart_predictions <- predict(final_wf_bart,
                                  new_data = ottotest,
                                  type = "prob") %>%
  bind_cols(., ottotest) %>% #Bind predictions with test data
  select(id, .pred_Class_1,.pred_Class_2,.pred_Class_3,.pred_Class_4,.pred_Class_5,
         .pred_Class_6,.pred_Class_7,.pred_Class_8,.pred_Class_9,) %>% 
  rename(Class_1=.pred_Class_1,Class_2=.pred_Class_2,Class_3=.pred_Class_3,
         Class_4=.pred_Class_4,Class_5=.pred_Class_5,Class_6=.pred_Class_6,Class_7=.pred_Class_7,
         Class_8=.pred_Class_8,Class_9=.pred_Class_9,)

vroom_write(x=otto_bart_predictions, file="./OTTOBARTPreds.csv", delim=",")

# random forest

# Classification Random Forests
my_mod_rf <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create a workflow with model and recipe


wf_rf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(my_mod_rf)
# Set up grid of tuning values
tuning_grid_rf <- grid_regular(mtry(range = c(2,94)),
                               min_n(),
                               levels = 5)

# Set up K-fold CV
folds <- vfold_cv(ottotrain, v = 5, repeats = 1)
# Run the CV
CV_results_rf <-wf_rf  %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_rf,
            metrics = metric_set(roc_auc))

# Find best tuning parameters
bestTune_rf <- CV_results_rf %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf_rf <- 
  wf_rf %>%
  finalize_workflow(bestTune_rf) %>%
  fit(data = ottotrain)

otto_rf_predictions <- predict(final_wf_rf,
                                 new_data = ottotest,
                                 type = "prob") %>%
  bind_cols(., ottotest) %>% #Bind predictions with test data
  select(id, .pred_Class_1,.pred_Class_2,.pred_Class_3,.pred_Class_4,.pred_Class_5,
         .pred_Class_6,.pred_Class_7,.pred_Class_8,.pred_Class_9,) %>% 
  rename(Class_1=.pred_Class_1,Class_2=.pred_Class_2,Class_3=.pred_Class_3,
         Class_4=.pred_Class_4,Class_5=.pred_Class_5,Class_6=.pred_Class_6,Class_7=.pred_Class_7,
         Class_8=.pred_Class_8,Class_9=.pred_Class_9,)

vroom_write(x=otto_rf_predictions, file="./OTTORFPreds.csv", delim=",")

#--------
#-- Medium: https://towardsdatascience.com/big-sales-mart-regression-revisited-enter-the-tidymodels-a6a432be58d4
#--------

#-- Data
# https://courses.analyticsvidhya.com/courses/big-mart-sales-prediction-using-r

#Load Packages
library(tidymodels)
library(magrittr) #Tidy pipes
library(patchwork) #Visualisation grids
library(stringr) #String Manipulation
library(forcats) #Working with Factors
library(corrr) #Develop tidy Correlation Plots
library(vip) #Most important feature visualisations
library(readr)
library(data.table)
library(skimr)

#Load Data
train <- fread("train.csv") %>% as.data.frame()
skim(train)


#--- EDA
viz_by_dtype <- function(combi, x, y ) {
  title <- str_replace_all(y, "_", " ") %>% 
           str_to_title()
  if ("factor" %in% class(x)) {
    ggplot(combi, aes(x, fill = x)) +
      geom_bar() +
      theme(legend.position = "none",
            axis.text.x = element_text(angle = 45, hjust = 1),
            axis.text = element_text(size = 8)) +
      theme_minimal() +
      scale_fill_viridis_d() +
      labs(title = title, y = "", x = "")
  }
  else if ("numeric" %in% class(x)) {
    ggplot(combi, aes(x)) +
      geom_histogram()  +
      theme_minimal() +
      scale_fill_viridis_d() +
      labs(title = title, y = "", x = "")
  } 
  else if ("integer" %in% class(x)) {
    ggplot(combi, aes(x)) +
      geom_histogram() +
      theme_minimal() +
      scale_fill_viridis_d() +
      labs(title = title, y = "", x = "")
  }
  else if ("character" %in% class(x)) {
    ggplot(combi, aes(x, fill = x)) +
      geom_bar() +
      theme_minimal() +
      scale_fill_viridis_d() +
      theme(legend.position = "none",
            axis.text.x = element_text(angle = 45, hjust = 1),
            axis.text = element_text(size = 8)) +
      labs(title = title, y = "", x = "")
  }
}

variable_list <- colnames(train) %>% as.list()
variable_plot <- map2(train, variable_list, viz_by_dtype) %>%
  wrap_plots(               
    ncol = 3,
    nrow = 4,
    heights = 150,
    widths = 150
  )

#-------------- EDA representation
#-- inspectdf
library(inspectdf)
var_num <- inspect_num(train)
show_plot(var_num)
var_cat <- inspect_cat(train)
show_plot(var_cat)
var_cor <- inspect_cor(train)
show_plot(var_cor)
#-- DataExplorer 
library(DataExplorer)
plot_histogram(train)
train %>%  
  plot_bar()


#--- Feature Engineering
#Correct mislabeled Item_Fat_Content
train %<>% mutate(Item_Fat_Content = if_else(Item_Fat_Content %in% c("reg", "Regular"), "Regular", "Low Fat"))
#Outlet Size is missing a label
train %<>% mutate(Outlet_Size = if_else(is.na(Outlet_Size),"Small",Outlet_Size))


#-- Data Splitting
set.seed(55)
mart_split <- initial_split(train, prop = 0.75, strata = Item_Outlet_Sales)
mart_train <- training(mart_split)
mart_test <- testing(mart_split)


#-- Recipe
mart_recipe <- 
  training(mart_split) %>% 
  recipe(Item_Outlet_Sales ~ .) %>% 
  step_rm(Item_Identifier) %>%
  step_impute_bag(Item_Weight) %>% 
  step_impute_knn(Item_Visibility) %>% 
  step_mutate(Price_Per_Unit = Item_MRP/Item_Weight) %>% 
  step_sqrt(Item_Visibility) %>%
  step_log(Price_Per_Unit, offset = 1) %>% 
  step_discretize(Item_MRP,num_breaks = 4) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal())

mart_recipe_prepped <- prep(mart_recipe)
  
#-- Train
mart_train <- bake(mart_recipe_prepped, new_data = mart_train)
dim(mart_train)
# [1] 6391 54

#-- Test
mart_test <- mart_recipe_prepped %>% 
             bake(testing(mart_split))
dim(mart_test)
# [1] 2132 54

#-- Correlation
corr_df <- mart_train %>% select(where(is.numeric)) %>% 
              correlate() %>%
              rearrange() %>% 
              shave()

rplot(corr_df,) +
 theme_minimal() +
 theme(axis.text.x = element_text(angle = 90)) +
 scale_colour_viridis_c()
 
#-- Modelling
#-- Linear
lm_model <- linear_reg() %>% 
            set_engine("lm")

lm_mart_fit <-   lm_model %>%
  fit(Item_Outlet_Sales ~ ., data = mart_train)

lm_mart_res <- 
  predict(lm_mart_fit, new_data = mart_test) %>% 
  bind_cols(mart_test %>% select(Item_Outlet_Sales))

lm_mart_res %>%
  ggplot(aes(x = Item_Outlet_Sales, y = .pred)) +
  geom_abline() +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(x = "Item Outlet Sales", y = "Predicted Item Outlet Sales")
  
#-- Metrics
metrics <- metric_set(rmse, rsq, mae)
metrics(lm_mart_res, truth = Item_Outlet_Sales, estimate = .pred)

#-- Workflow + ranger
rf_workflow <- 
  workflow() %>% 
  add_recipe(mart_recipe) %>% #Pre-processing Steps
  add_model(rand_forest(mode = "regression") %>% #Specify Model
              set_engine("ranger"))
              

rf_mart_res <- 
  rf_workflow %>% 
  fit(training(mart_split)) %>% 
  predict(new_data = testing(mart_split)) %>%
  bind_cols(mart_test %>% select(Item_Outlet_Sales))
  
rf_mart_res %>% 
  ggplot(aes(x = Item_Outlet_Sales, y = .pred)) +
   geom_abline(lty = 2) +
   geom_point(alpha = 0.5) +
   theme_minimal() +
   labs(x = "Item Outlet Sales", y = "Predicted Item Outlet Sales", title = "pRandom Forest Regression")
   

#-- Prevent overfitting by randomForest with Dials.
set.seed(256)
rf_mod <- 
  rand_forest(trees = 500,
              mtry = tune(),
              min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity", num.threads = 6) %>% 
  set_mode("regression")#Establish Model Flow
tune_wf <- workflow() %>%
  add_recipe(mart_recipe) %>%
  add_model(rf_mod)
  
#Generate grid to perform grid search for hyperparameter optimisation 
rf_grid <- grid_regular(mtry(range = c(6,10)), 
                        min_n(range = c(14,20)), 
                        levels = c(10,9))

# 4-fold Cross Validation Stratified by Item_Outlet_Sales
folds <- vfold_cv(train, v = 4, strata = Item_Outlet_Sales)

#Train and evaluate all combinations of hyperparameters specified in rf_grid
doParallel::registerDoParallel(cores = 4)
rf_grid_search <- tune_grid(
  tune_wf,
  resamples = folds,
  grid = rf_grid)

#-- Visualize parameters.
rf_grid_search %>%
  collect_metrics() %>% 
  filter(.metric == "rmse") %>%
  select(mean, min_n, mtry) %>%
  filter(mtry > 4) %>% 
  ggplot(aes(min_n, mean, color = as_factor(mtry))) +
  geom_point()  +
  geom_line() +
  scale_color_viridis_d() +
  theme_minimal() +
  scale_x_continuous(breaks = pretty_breaks()) +
  theme(legend.position = "bottom") +
  labs(x = "Minimum Number of Observations to Split Node", y = "RMSE", title = "Grid Search Results for Random Forest", color = "Number of Predictors Sampled at Each Split")

#-- Best Model
rf_grid_search %>% show_best()

#-- For the best calculate rmse
rf_best_rmse <- select_best(rf_grid_search, "rmse")
final_rf <- finalize_model(
  rf_mod,
  rf_best_rmse
)
final_rf


#-- Important Variables
final_rf %>%
  fit(Item_Outlet_Sales ~., data = bake(prep(mart_recipe), training(mart_split))) %>% 
  vip(geom = c("col"), num_features = 10) +
  theme_minimal()

#--- Final model
final_wf <- 
  workflow() %>% 
  add_recipe(mart_recipe) %>% 
  add_model(final_rf)
  
final_rf_res <- 
  fit(final_wf, training(mart_split)) %>% 
  predict(new_data = testing(mart_split)) %>% 
  bind_cols(mart_test %>% select(Item_Outlet_Sales))
  
final_rf_res %>% ggplot(aes(x = Item_Outlet_Sales, y = .pred)) +
                  geom_abline(lty = 2) +
                  geom_point(alpha = 0.5) +
                  theme_minimal() +
                  labs(x = "Item Outlet Sales", 
                       y = "Predicted Item Outlet Sales", 
                       title = "Tuned Random Forest Regression")
                  
metrics(final_rf_res, truth = Item_Outlet_Sales, estimate = .pred)



if (!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)
if (!require(dplyr)) install.packages("dplyr")
library(dplyr)
if (!require(tidyverse)) install.packages("tidyverse")
library(tidyverse)
if (!require(stargazer)) install.packages("stargazer")
library(stargazer)
if (!require(rnaturalearth)) install.packages("rnaturalearth")
library(rnaturalearth)
if (!require(maps)) install.packages("maps")
library(maps)
if (!require(sf)) install.packages("sf")
library(sf)
if (!require(glmnet)) install.packages("glmnet")
library(glmnet)
if (!require(caret)) install.packages("caret")
library(caret)
if (!require(forecast)) install.packages("forecast")
library(forecast)
if (!require(TTR)) install.packages("TTR")
library(TTR)
if (!require(fpp2)) install.packages("fpp2")
library(fpp2)

rm(list = ls())
set.seed(123)
dir <- getwd()

setwd(file.path(dir, "..", "Data"))

QOL_df <- read.csv("QOL.csv")
WHI_df <- read.csv("WHI.csv")
GPI_df <- read.csv("GPI.csv")
head(QOL_df)

colnames(QOL_df) <- tolower(colnames(QOL_df))
colnames(WHI_df) <- tolower(colnames(WHI_df))
colnames(GPI_df) <- tolower(colnames(GPI_df))


merged_df <- merge(WHI_df,GPI_df, by = c("year", "country") , all = TRUE)
merged_all <- merge(merged_df,QOL_df, by = c("year", "country"), all = TRUE)
merged_all <- na.omit(merged_all)

merged_all$rank.x <- NULL
merged_all$rank.y <- NULL
rownames(merged_all) <- NULL
colnames(merged_all)

category <- c("year","country","access.to.weapons","neighbouring.country.relations","organised.conflict..internal.", "political.instability","political.terror","violent.crime","violent.demonstrations")
category_df <-  merged_all[, names(merged_all) %in% category]
category_df[,3:ncol(category_df)] <- as.data.frame(lapply(category_df[,3:ncol(category_df)],factor))
numerical_df1 <- select(merged_all, -c("year","country","access.to.weapons","neighbouring.country.relations","organised.conflict..internal.", "political.instability","political.terror","violent.crime","violent.demonstrations"))

numerical_df1$climate.index <- sub('_', 74.397, numerical_df1$climate.index)
numerical_df1$climate.index <- as.numeric(numerical_df1$climate.index)


#Scaling Function
min_max_scaling <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}
numerical_df1 <- numerical_df1 %>% mutate(cumulative_index = life.ladder+global.average+ quality.of.life.index )
scaled_dataset <- as.data.frame(lapply(numerical_df1, min_max_scaling))

standardized_data_df <- as.data.frame(scale(scaled_dataset, center = T))


standardized_data_df <- cbind(category_df, standardized_data_df)
standardized_data_df_full <- standardized_data_df
original_data <- standardized_data_df_full
original_data <- original_data[-318,]


columns_to_remove <- c("year","country","life.ladder","global.average","quality.of.life.index")
#final_df <- select(scaled_data, -c("year","country","life.ladder","global.average","quality.of.life.index"))
standardized_data_df <- select(standardized_data_df, -c("year","country","life.ladder","global.average","quality.of.life.index"))
model <- lm(cumulative_index~., data = standardized_data_df)
summary(model)
standardized_data_df <- standardized_data_df[-318,]
model <- lm(cumulative_index~., data = standardized_data_df)
summary(model)
plot(model)


##------------------------------- Insights --------------------------------######

gdp_cumulative <- lm(cumulative_index ~ log.gdp.per.capita , data = standardized_data_df)

# Print the summary
summary(gdp_cumulative)
stargazer(gdp_cumulative, type = 'text')

# Assuming your data frame is named 'standardized_data_df'
pollution_gdp <- lm(log.gdp.per.capita ~ pollution.index, data = standardized_data_df)
summary(pollution_gdp)
stargazer(pollution_gdp, type = 'text')




# Create a ggplot scatter plot
ggplot(scaled_dataset, aes(x = log.gdp.per.capita, y = pollution.index)) +
  geom_point(color = "blue", size = 3) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +  # Add a trendline
  
  # Customize the theme and labels
  theme_minimal() +
  labs(title = "Scatter Plot of Log GDP per Capita vs Pollution Index",
       x = "Log GDP per Capita", y = "Pollution Index") +
  
  # Customize the axis breaks
  scale_x_continuous(breaks = seq(0, 1, by = 0.2), labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2), labels = seq(0, 1, by = 0.2)) +
  
  # Add a legend
  theme(legend.position = "topright") +
  guides(color = guide_legend(title = "Trendline"))


healthcare_gdp <- lm(log.gdp.per.capita ~ health.care.index , data = standardized_data_df)
summary(healthcare_gdp)
stargazer(healthcare_gdp, type = "text")



# Create a ggplot scatter plot
ggplot(scaled_dataset, aes(x = log.gdp.per.capita, y = health.care.index)) +
  geom_point(color = "blue", size = 3) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +  # Use lm directly
  
  # Customize the theme and labels
  theme_minimal() +
  labs(title = "Scatter Plot of Log GDP per Capita vs Health Care Index",
       x = "Log GDP per Capita", y = "Health Care Index") +
  
  # Customize the axis breaks
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  
  # Add a legend
  theme(legend.position = "topright") +
  guides(color = guide_legend(title = "Trendline"))

traffic_gdp <- lm(log.gdp.per.capita ~ traffic.commute.time.index , data = standardized_data_df)
summary(traffic_gdp)
stargazer(traffic_gdp, type = "text")

ggplot(scaled_dataset, aes(x = log.gdp.per.capita, y = traffic.commute.time.index)) +
  geom_point(color = "blue", size = 3) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +  # Use lm directly
  
  # Customize the theme and labels
  theme_minimal() +
  labs(title = "Scatter Plot of Log GDP per Capita vs Traffic Commute Time",
       x = "Log GDP per Capita", y = "Traffic Commute Time") +
  
  # Customize the axis breaks
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  
  # Add a legend
  theme(legend.position = "topright") +
  guides(color = guide_legend(title = "Trendline"))

hleab_gdp <- lm(log.gdp.per.capita ~ healthy.life.expectancy.at.birth , data = standardized_data_df)
summary(hleab_gdp)
stargazer(hleab_gdp, type = "text")

ggplot(scaled_dataset, aes(x = log.gdp.per.capita, y = healthy.life.expectancy.at.birth)) +
  geom_point(color = "blue", size = 3) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +  # Use lm directly
  
  # Customize the theme and labels
  theme_minimal() +
  labs(title = "Scatter Plot of Log GDP per Capita vs Healthy Life (Mental Health) Expectancy",
       x = "Log GDP per Capita", y = "Healthy Life Expectancy at Birth") +  # Corrected label
  
  # Customize the axis breaks
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  
  # Add a legend
  theme(legend.position = "topright") +
  guides(color = guide_legend(title = "Trendline"))

###---------------------Geo-Spatial---------------------------------------###

geoplot_df <- read.csv("geoplot_df.csv")

# Get world map data
world_map <- ne_countries(scale = "medium", returnclass = "sf")

# Get continent names
continents <- ne_download(category = "cultural", type = "continent", returnclass = "sf")

# Merge the world map data with your data
merged_data <- merge(world_map, geoplot_df, by.x = "name", by.y = "country", all.x = TRUE)

color_scale <- scale_fill_gradient(low = "lightblue", high = "darkblue")

# Plot the map using ggplot2
ggplot() +
  geom_sf(data = merged_data, aes(fill = cumulative_index, geometry = geometry), color = "white") +
  color_scale +
  labs(title = "Cumulative Index by Country") +
  theme_minimal()
df <- merged_data %>% select(name, continent, cumulative_index)

df <- df %>% filter(!is.na(cumulative_index))
# Convert 'sf' data frame to a regular data frame
df_regular <- as.data.frame(df)

# Calculate the correlation matrix
correlation_matrix <- cor(df_regular[, "cumulative_index", drop = FALSE])


# Pairwise Comparisons
df$cumulative_index <- as.numeric(df$cumulative_index)


# Grouping by Continent
continent_means <- df_regular %>%
  group_by(continent) %>%
  summarize(mean_cumulative_index = mean(cumulative_index))
continent_means

# Plotting Pairwise Comparisons
ggplot(df_regular, aes(x = cumulative_index, y = name)) +
  geom_point() +
  labs(title = "Pairwise Comparison of Cumulative Index",
       x = "Cumulative Index",
       y = "Country") +
  theme_minimal()

#######-----------------------Utopia------------------------------------------##


utopia <- read.csv("scaled_dataset_utopia.csv")
utopia <- utopia %>% filter(year == 2022)
filtered <- utopia %>% filter (country == "Utopia" | country == "Switzerland" | country == "Iran" | country == "Mexico" )
filtered <- filtered %>% select(-year)

df_long <- reshape2::melt(filtered, id.vars = "country")

colors <- c("Switzerland" = "gray", "Utopia" = "blue", "Iran" = "lightgray", "Mexico" = "darkgray")


# Columns to display data labels
index_columns <- c(
  "un.peacekeeping.funding", "purchasing.power.index", "safety.index",
  "health.care.index", "cost.of.living.index", "property.price.to.income.ratio",
  "traffic.commute.time.index", "pollution.index", "climate.index", "cumulative_index"
)


# Plotting with tilted x-axis labels and value labels
ggplot(df_long, aes(x = variable, y = value, fill = country, label = ifelse(variable %in% index_columns, round(value, 1), ""))) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(position = position_dodge(width = 0.9), vjust = -0.5) +
  scale_fill_manual(values = colors) +
  labs(title = "Comparison of Index Values for Iran, Switzerland and Mexico vs Utopia",
       y = "Index Value",  x = "")+
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, face = "bold", size = 12))  # Adjust the size as needed

########## FEATURE SELECTION ##########




final_df_filtered <- standardized_data_df
head(final_df_filtered)


# ---------------------------- Stepwise Regression -------------------------------------

# Stepwise Regression using original variables and Cross Validation
# In backward stepwise regression. Our lower model will have only the intercept
# and all variables in our full model.



# Now using the code below to perform 5 fold CV

ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

lmFit_Step <- train(cumulative_index ~ ., data = final_df_filtered, "lmStepAIC", scope = 
                      list(lower = cumulative_index~1, upper = cumulative_index~.), direction = "backward",trControl=ctrl)




#Fitting a new model with these 36 variables.

mod_Step = lm(cumulative_index ~ access.to.weapons + neighbouring.country.relations + 
                organised.conflict..internal. + political.instability + political.terror + 
                violent.crime + violent.demonstrations + log.gdp.per.capita + social.support + healthy.life.expectancy.at.birth + 
                freedom.to.make.life.choices + generosity + perceptions.of.corruption + 
                positive.affect + negative.affect +  
                deaths.from.conflict..external. + deaths.from.conflict..internal. + 
                displaced.people + domestic...international.conflict + external.conflicts.fought + 
                internal.conflicts.fought + jailed.population + military.expenditure + safety...security + 
                security.officers...police + terrorist.activity + 
                weapons.exports + purchasing.power.index + safety.index + 
                health.care.index + cost.of.living.index + property.price.to.income.ratio + 
                traffic.commute.time.index + pollution.index + climate.index, data = final_df_filtered)
summary(mod_Step)


#We obtain an Adjusted R-SQuared value = 0.9999 using the selected 36 variables.
# Let's rerun a regression with only the most important variables (26) from the previous regression. 



#Fitting a new model with these 28 variables.

mod_Step = lm(cumulative_index ~ access.to.weapons +  
                political.instability + political.terror + 
                violent.crime + violent.demonstrations + log.gdp.per.capita + social.support + healthy.life.expectancy.at.birth + 
                freedom.to.make.life.choices + generosity + perceptions.of.corruption + 
                positive.affect + negative.affect + 
                displaced.people +  
                jailed.population + military.expenditure + safety...security + 
                security.officers...police + terrorist.activity + 
                weapons.exports + purchasing.power.index + safety.index + 
                health.care.index + cost.of.living.index + property.price.to.income.ratio + 
                traffic.commute.time.index + pollution.index + climate.index, data = final_df_filtered)
summary(mod_Step)


#We obtain an Adjusted R-SQuared value = 0.9999 using the selected 28 variables.






# ---------------------------- Lasso Regression -------------------------------------


#building lasso

XP=data.matrix(final_df_filtered[,-43])
YP=data.matrix(final_df_filtered$cumulative_index)
lasso=cv.glmnet(x=as.matrix(final_df_filtered[,-43]),y=as.matrix(final_df_filtered$cumulative_index),alpha=1,
                nfolds = 5,type.measure="mse",family="gaussian")

#Output the coefficients of the variables selected by lasso

coef(lasso, s=lasso$lambda.min)



#Fitting a new model with these 9 variables.

mod_lasso = lm(cumulative_index ~un.peacekeeping.funding + purchasing.power.index + safety.index +
                 health.care.index + cost.of.living.index + property.price.to.income.ratio +
                 traffic.commute.time.index + pollution.index + climate.index, data = final_df_filtered)
summary(mod_lasso)

plot(mod_lasso, ask=FALSE)


# ---------------------------- Elastic Net -------------------------------------

#We vary alpha in steps of 0.1 from 0 to 1 and calculate the resultant R-Squared values

R2=c()
for (i in 0:10) {
  mod_elastic = cv.glmnet(x=as.matrix(final_df_filtered[,-43]),y=as.matrix(final_df_filtered$cumulative_index),
                          alpha=i/10,nfolds = 5,type.measure="mse",family="gaussian")
  

  
  R2 = cbind(R2,mod_elastic$glmnet.fit$dev.ratio[which(mod_elastic$glmnet.fit$lambda == mod_elastic$lambda.min)])
  
}

R2


#Best value of alpha

alpha_best = (which.max(R2)-1)/10
alpha_best

## 0.9


#Lets build the model using this alpha value.

Elastic_net=cv.glmnet(x=as.matrix(final_df_filtered[,-43]),y=as.matrix(final_df_filtered$cumulative_index),alpha=alpha_best,
                      nfolds = 5,type.measure="mse",family="gaussian")

#Output the coefficients of the variables selected by Elastic Net

coef(Elastic_net, s=Elastic_net$lambda.min)



# The Elastic Net selects 11 variables. Next we compare how this new model performs 

#Fitting a new model with these 11 variables.

mod_Elastic_net = lm(cumulative_index ~ social.support + freedom.to.make.life.choices +
                       un.peacekeeping.funding + purchasing.power.index + safety.index + 
                       health.care.index + cost.of.living.index + property.price.to.income.ratio +
                       traffic.commute.time.index + pollution.index + climate.index, data = final_df_filtered)
summary(mod_Elastic_net)

#We obtain an Adjusted R-SQuared value = 0.9997 using the selected 11 variables.

plot(mod_Elastic_net, ask=FALSE)



#####################K-means clustering############################################
set.seed(123)
## In order to find the optimal number of clusterings, we decide to use elbow method which looks at the 
## percentage of explained variance as a function of the number of clusters.
## Use elbow method to find the optinal number of clusters
k_max <- 10
num_data_df <- standardized_data_df

wss <- sapply(1:k_max, function(k) {
  kmeans(standardized_data_df, centers = k, nstart = 50)$tot.withinss
})
plot(1:k_max, wss, type = "b", xlab = "Number of Clusters", ylab = "Total within-cluster sum of squares")
##Looking at the graph, we can see that the wss has decreased significantly as the number of clusters increase from 1 to 2.
##Furthermore, we can also see a slight "elbow" bend starting from k = 3 so it is a good starting point to select for our optimal clusters
## Now we run the k means clustering using k = 3:

set.seed(123)
km <- kmeans(num_data_df, centers = 3, nstart = 20)
countries <- original_data$country
## After getting our clusters, we want to add back the country and year column
#original_data <- standardized_data_df_full
consolidated_clusters <- data.frame(
  country = original_data$country,
  year = original_data$year,
  cluster = km$cluster
)
consolidated_clusters

## Tracking total number of each cluster by year:
cluster_counts <- consolidated_clusters %>%
  group_by(year, cluster) %>%
  summarize(Count = n(), .groups = 'drop') %>%
  print(n = Inf)
consolidated_clusters$year <- as.factor(consolidated_clusters$year)

ggplot(cluster_counts, aes(x = year, y = Count, group = cluster, color = as.factor(cluster))) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = unique(cluster_counts$year)) + # Set x-axis breaks
  theme_minimal() +
  labs(title = "Cluster Count by Year", x = "Year", y = "Count", color = "Cluster")
## Countries by cluster on Worldmap by year:
world <- ne_countries(scale = "medium", returnclass = "sf")
map_data <- merge(world, consolidated_clusters, by.x = "name", by.y = "country")
ggplot() + 
  geom_sf(data = map_data, aes(fill = factor(cluster), geometry = geometry)) +
  facet_wrap(~ year) +  # Creates a separate map for each year
  scale_fill_brewer(palette = "Set1") +  # Color palette for clusters
  labs(fill = "Cluster") + 
  theme_minimal() +
  ggtitle("World Map of Clusters Over Years")

#######---------------------TimeSeries Forecasting ------------------------------

scaled_dataset <- cbind(category_df, scaled_dataset)


scaled_data <- scaled_dataset
df <- scaled_data %>% select(year, country, cumulative_index)


# last two years data 21 & 22 as testing data, create a test and train dataset
train <- filter(df, df$year <= 2020)
test <- filter(df, df$year > 2020)

# Iran
train_iran <- filter(train, train$country == "Iran")
iran_ts<- ts(train_iran$cumulative_index)
plot.ts(iran_ts)
test_iran <- filter(test, test$country == "Iran")

#SES() - useful for data w/ no trend or seasonality
ses_iran <- ses(train_iran$cumulative_index, h = 2)
accuracy(ses_iran, test_iran$cumulative_index)

#Exponential Smoothing w/ Trend
holt_iran <- holt(train_iran$cumulative_index, h = 2)
accuracy(holt_iran, test_iran$cumulative_index)

#ETS - error, trend, smoothing
ets_iran <- ets(train_iran$cumulative_index)
ets_fc_iran <- forecast(ets_iran, h=2)
accuracy(ets_fc_iran, test_iran$cumulative_index)

#ARIMA:
# Create arima model 0,0,2 from auto.arima optimizer
arima_iran = arima(train_iran$cumulative_index, order = c(0,0,2))
# Forecast out 2 years
arima_fc_iran = forecast(arima_iran, h=2)
accuracy(arima_fc_iran, test_iran$cumulative_index)



# Mexico
train_Mexico <- filter(train, train$country == "Mexico")
Mexico_ts<- ts(train_Mexico$cumulative_index)
plot.ts(Mexico_ts)
test_Mexico <- filter(test, test$country == "Mexico")

#SES() - useful for data w/ no trend or seasonality
ses_Mexico <- ses(train_Mexico$cumulative_index, h = 2)
accuracy(ses_Mexico, test_Mexico$cumulative_index)

#Exponential Smoothing w/ Trend
holt_Mexico <- holt(train_Mexico$cumulative_index, h = 2)
accuracy(holt_Mexico, test_Mexico$cumulative_index)

#ETS - error, trend, smoothing
ets_Mexico <- ets(train_Mexico$cumulative_index)
ets_fc_Mexico <- forecast(ets_Mexico, h=2)
accuracy(ets_fc_Mexico, test_Mexico$cumulative_index)

#ARIMA:
# Create arima model 0,0,2 from auto.arima optimizer
arima_Mexico = arima(train_Mexico$cumulative_index, order = c(0,0,2))
# Forecast out 2 years
arima_fc_Mexico = forecast(arima_Mexico, h=2)
accuracy(arima_fc_Mexico, test_Mexico$cumulative_index)


# Switzerland
train_Switzerland <- filter(train, train$country == "Switzerland")
Switzerland_ts<- ts(train_Switzerland$cumulative_index)
plot.ts(Switzerland_ts)
test_Switzerland <- filter(test, test$country == "Switzerland")

#SES() - useful for data w/ no trend or seasonality
ses_Switzerland <- ses(train_Switzerland$cumulative_index, h = 2)
accuracy(ses_Switzerland, test_Switzerland$cumulative_index)

#Exponential Smoothing w/ Trend
holt_Switzerland <- holt(train_Switzerland$cumulative_index, h = 2)
accuracy(holt_Switzerland, test_Switzerland$cumulative_index)

#ETS - error, trend, smoothing
ets_Switzerland <- ets(train_Switzerland$cumulative_index)
ets_fc_Switzerland <- forecast(ets_Switzerland, h=2)
accuracy(ets_fc_Switzerland, test_Switzerland$cumulative_index)

#ARIMA:
# Create arima model 0,0,2 from auto.arima optimizer
arima_Switzerland = arima(train_Switzerland$cumulative_index, order = c(0,0,2))
# Forecast out 2 years
arima_fc_Switzerland = forecast(arima_Switzerland, h=2)
accuracy(arima_fc_Switzerland, test_Switzerland$cumulative_index)
autoplot(arima_fc_iran)
autoplot(arima_fc_Mexico)
autoplot(arima_fc_Switzerland)

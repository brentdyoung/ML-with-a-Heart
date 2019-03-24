#Brent Young
#MSDS 454
#Summer 2018
#Final Project

#Load Libraries

library(boot) #bootstrap
library(leaps) #Best subset selection; stepwise
library(glmnet) #Ridge Regression & the Lasso
library(pls) #Principal components Regression
library(splines) #regression splines
library(gam) #Generalized Additive Models
library(akima)
library(caret) #Machine Learning
library(yardstick) #Machine Learning
library(class) #KNN
library(nnet) # Neural Network
library(tree) # Decision Trees 
library(randomForest) # Bagging and Random Forest
library(e1071) #Support Vector Machines, Naive Bayes Classifier
library(gbm) # Gradient Boosting Machines
library(xgboost) # Gradient Boosting Machines
library(mda) #Mixture Discriminant Analysis

library(Hmisc)
library(VIM) #Missingness Map
library(pROC) #ROC Curve
library(ROCR) #ROC Curve, AUC
library(AUC)
library(corrplot) #Data Visualization
library(ggcorrplot) #Data Visualization
library (ggplot2) #Data Visualization
library(GGally) #Data Visualization
library(lattice)#Data Visualization
library(gridExtra)
library(gmodels)#Cross-tabs
library(dplyr)
library(reshape2)
library(readr)
library(glm2)
library(aod)
library(rcompanion) 
library(leaps) #Best subsets
library(MASS) #Linear Discriminant Analysis
library(car)
library(rockchalk)

# Multiple plot function

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


#Load the data

setwd("~/R/MSDS 454 FINAL")

raw_train_values<- read.csv("train_values.csv")
raw_train_labels<- read.csv("train_labels.csv")
raw_train <- merge(raw_train_values,raw_train_labels,by="patient_id")

raw_test <- read.csv("test_values.csv")
raw_test["heart_disease_present"] <-NA
str(raw_test)

total <- rbind(raw_train,raw_test)

names(total) <- make.names(names(total)) 
colnames(total)[15] <- "TARGET_FLAG"       # rename Target

#Descriptive Statistics
str(total)
summary(total) # No N/A's, except 90 N/As in test set as expected
describe(total)
dim(total)

#Feature Creation
total <- total %>% mutate(max_predicted_heart_rate_achieved = max_heart_rate_achieved/(220-age)) # Percentage of Max. Predicted Heart Rate Achieved

#age_bin
total$age_bin[total$age >= 29 & total$age <= 44] <- "29 to 44"
total$age_bin[total$age >= 45 & total$age <= 64] <- "45 to 64"
total$age_bin[total$age >= 65] <- "65+"
total$age_bin <- factor(total$age_bin)
total$age_bin <- factor(total$age_bin, levels=c("29 to 44","45 to 64","65+"))

#Factor Variables
total$TARGET_FLAG <- as.factor(total$TARGET_FLAG)
total$fasting_blood_sugar_gt_120_mg_per_dl <- as.factor(total$fasting_blood_sugar_gt_120_mg_per_dl)
total$sex  <- as.factor(total$sex)
total$exercise_induced_angina  <- as.factor(total$exercise_induced_angina )

#Extras
#total <- total %>% mutate(MPHRA_100 = 1*(220-age))
#total <- total %>% mutate(MPHRA_85 = 0.85*(220-age))

#MPHRA_85_Exam
#total$MPHRA_85_Exam[total$max_heart_rate_achieved >= total$MPHRA_85] <- "1"
#total$MPHRA_85_Exam[total$max_heart_rate_achieved < total$MPHRA_85] <- "0"
#total$MPHRA_85_Exam <- factor(total$MPHRA_85_Exam, levels=c("1","0"))

#str(total)

#One-hot encoding
#for(unique_value in unique(total$thal)){
  #total[paste("thal", unique_value, sep = ".")] <- ifelse(total$thal == unique_value, 1, 0)
#}

#total <- total %>% mutate(rpp = max_heart_rate_achieved*resting_blood_pressure) # Rate Pressure Product	

################################# Quick High Level EDA for Overall Dataset #################################

#Descriptive Statistics
str(total)
summary(total) # No N/A's, except 90 N/As in test set as expected
describe(total)
dim(total)

#Check for Missingness
sapply(total, function(x) sum(is.na(x)))
sum(is.na(total))
aggr_plot <- aggr(total, col=c('#9ecae1','#de2d26'), numbers=TRUE,prop=FALSE, sortVars=TRUE, labels=names(total), cex.axis=.5, gap=2, ylab=c("Histogram of missing data","Pattern"))

####################################### Split Datasets ######################################
#Split Data into Train/Validation/Test 

train_set = total[1:180,]

#Recode the class labels to Yes/No (required when using class probs)
#train_set$TARGET_FLAG<-factor(train_set$TARGET_FLAG,levels = c(0,1),labels=c("No", "Yes"))
train_set$TARGET_FLAG <- as.factor(train_set$TARGET_FLAG)

str(train_set)

set.seed(123)
validationIndex <- createDataPartition(train_set$TARGET_FLAG, p=0.50, list=FALSE)

train <- train_set[validationIndex,]
validation <- train_set[-validationIndex,]
test = total[181:270,]

str(train)
summary(train)
describe(train)
aggr_plot <- aggr(train, col=c('#9ecae1','#de2d26'), numbers=TRUE,prop=FALSE, sortVars=TRUE, labels=names(train), cex.axis=.5, gap=2, ylab=c("Histogram of missing data","Pattern"))


str(validation)

str(test)
describe(test)


############################## EDA for Classification Models - Training Data ##############################

######EDA for Numeric Variables#####

par(mfrow=c(3,3))
hist(train$slope_of_peak_exercise_st_segment, col = "#ece7f2", xlab = "slope_of_peak_exercise_st_segment", main = "Histogram of slope_of_peak_exercise_st_segment")
hist(train$resting_blood_pressure , col = "#a6bddb", xlab = "resting_blood_pressure ", main = "Histogram of resting_blood_pressure ")
hist(train$chest_pain_type , col = "#3182bd", xlab = "chest_pain_type ", main = "Histogram of chest_pain_type ")
boxplot(train$slope_of_peak_exercise_st_segment, col = "#ece7f2", main = "Boxplot of slope_of_peak_exercise_st_segment")
boxplot(train$resting_blood_pressure , col = "#a6bddb", main = "Boxplot of resting_blood_pressure")
boxplot(train$chest_pain_type , col = "#3182bd", main = "Boxplot of chest_pain_type ")
par(mfrow=c(1,1))

par(mfrow=c(3,3))
hist(train$num_major_vessels, col = "#ece7f2", xlab = "num_major_vessels ", main = "Histogram of num_major_vessels ")
hist(train$resting_ekg_results  , col = "#a6bddb", xlab = "resting_ekg_results  ", main = "Histogram of resting_ekg_results  ")
hist(train$serum_cholesterol_mg_per_dl , col = "#3182bd", xlab = "serum_cholesterol_mg_per_dl ", main = "Histogram of serum_cholesterol_mg_per_dl ")
boxplot(train$num_major_vessels , col = "#ece7f2", main = "num_major_vessels ")
boxplot(train$resting_ekg_results  , col = "#a6bddb", main = "Boxplot of resting_ekg_results ")
boxplot(train$serum_cholesterol_mg_per_dl , col = "#3182bd", main = "Boxplot of serum_cholesterol_mg_per_dl ")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(train$oldpeak_eq_st_depression , col = "#ece7f2", xlab = "oldpeak_eq_st_depression  ", main = "Histogram of oldpeak_eq_st_depression  ")
hist(train$age   , col = "#a6bddb", xlab = "age   ", main = "Histogram of age   ")
boxplot(train$oldpeak_eq_st_depression  , col = "#ece7f2", main = "oldpeak_eq_st_depression  ")
boxplot(train$age   , col = "#a6bddb", main = "Boxplot of age  ")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(train$max_heart_rate_achieved  , col = "#3182bd", xlab = "max_heart_rate_achieved  ", main = "Histogram of max_heart_rate_achieved")
hist(train$max_predicted_heart_rate_achieved, col = "#ece7f2", xlab = "max_predicted_heart_rate_achieved  ", main = "Histogram of max_predicted_heart_rate_achieved")
boxplot(train$max_heart_rate_achieved  , col = "#3182bd", main = "Boxplot of max_heart_rate_achieved  ")
boxplot(train$max_predicted_heart_rate_achieved  , col = "#ece7f2", main = "max_predicted_heart_rate_achieved")
par(mfrow=c(1,1))

#Boxplots for Numeric Variables
#a2<-ggplot(data = train, aes(x = "", y = slope_of_peak_exercise_st_segment )) + 
#geom_boxplot(fill='#3182bd', color="black")
#a4<-ggplot(data = train, aes(x = "", y = resting_blood_pressure )) + 
#geom_boxplot(fill='#3182bd', color="black")
#a6<-ggplot(data = train, aes(x = "", y = chest_pain_type )) + 
#geom_boxplot(fill='#3182bd', color="black")
#b2<-ggplot(data = train, aes(x = "", y = num_major_vessels )) + 
#geom_boxplot(fill='#3182bd', color="black")
#b4<-ggplot(data = train, aes(x = "", y = resting_ekg_results)) + 
#geom_boxplot(fill='#3182bd', color="black")
#b6<-ggplot(data = train, aes(x = "", y = serum_cholesterol_mg_per_dl )) + 
#geom_boxplot(fill='#3182bd', color="black")
#c2<-ggplot(data = train, aes(x = "", y = oldpeak_eq_st_depression )) + 
#geom_boxplot(fill='#3182bd', color="black")
#c4<-ggplot(data = train, aes(x = "", y = age )) + 
#geom_boxplot(fill='#3182bd', color="black")
#c6<-ggplot(data = train, aes(x = "", y = max_heart_rate_achieved )) + 
#geom_boxplot(fill='#3182bd', color="black")
#c8<-ggplot(data = train, aes(x = "", y = max_predicted_heart_rate_achieved )) + 
#geom_boxplot(fill='#3182bd', color="black")

#Histogram, Density, and Boxplots
#Histogram & Density of slope_of_peak_exercise_st_segment
a1<-ggplot(train, aes(x=slope_of_peak_exercise_st_segment)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(slope_of_peak_exercise_st_segment, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("slope_of_peak_exercise_st_segment") +
  theme(plot.title = element_text(hjust = 0.5))

a2<-ggplot(train, aes(x=TARGET_FLAG, y= slope_of_peak_exercise_st_segment )) + 
  geom_boxplot(fill="#bdc9e1",notch=TRUE) +
  labs(title="Distribution of slope_of_peak_exercise_st_segment ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#Histogram & Density of resting_blood_pressure
a3<-ggplot(train, aes(x=resting_blood_pressure)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(resting_blood_pressure, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("resting_blood_pressure") +
  theme(plot.title = element_text(hjust = 0.5))

a4<-ggplot(train, aes(x=TARGET_FLAG, y= resting_blood_pressure )) + 
  geom_boxplot(fill="#3182bd",notch=TRUE) +
  labs(title="Distribution of resting_blood_pressure ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#Histogram & Density of chest_pain_type
a5<-ggplot(train, aes(x=chest_pain_type)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(chest_pain_type, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("chest_pain_type") +
  theme(plot.title = element_text(hjust = 0.5))

a6<-ggplot(train, aes(x=TARGET_FLAG, y= chest_pain_type )) + 
  geom_boxplot(fill="#f1eef6",notch=TRUE) +
  labs(title="7a: chest_pain_type") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

multiplot(a1, a2, a3, a4, a5, a6, cols=3)


#Histogram & Density of num_major_vessels
b1<-ggplot(train, aes(x=num_major_vessels)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(num_major_vessels, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("num_major_vessels") +
  theme(plot.title = element_text(hjust = 0.5))

b2 <-ggplot(train, aes(x=TARGET_FLAG, y= num_major_vessels )) + 
  geom_boxplot(fill="#bdc9e1",notch=TRUE) +
  labs(title="7b: num_major_vessels") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))


#Histogram & Density of resting_ekg_results
b3<-ggplot(train, aes(x=resting_ekg_results)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(resting_ekg_results, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("resting_ekg_results") +
  theme(plot.title = element_text(hjust = 0.5))

b4 <-ggplot(train, aes(x=TARGET_FLAG, y= resting_ekg_results )) + 
  geom_boxplot(fill="#3182bd", notch=TRUE) +
  labs(title="Distribution of resting_ekg_results ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))


#Histogram & Density of serum_cholesterol_mg_per_dl
b5<-ggplot(train, aes(x=serum_cholesterol_mg_per_dl)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(serum_cholesterol_mg_per_dl, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("serum_cholesterol_mg_per_dl") +
  theme(plot.title = element_text(hjust = 0.5))

b6<-ggplot(train, aes(x=TARGET_FLAG, y= serum_cholesterol_mg_per_dl )) + 
  geom_boxplot(fill="#74a9cf",notch=TRUE) +
  labs(title="7c: serum_cholesterol_mg_per_dl") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

multiplot(b1, b2, b3, b4, b5, b6, cols=3)

#Histogram & Density of oldpeak_eq_st_depression
c1<-ggplot(train, aes(x=oldpeak_eq_st_depression)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(oldpeak_eq_st_depression, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("oldpeak_eq_st_depression") +
  theme(plot.title = element_text(hjust = 0.5))

c2<-ggplot(train, aes(x=TARGET_FLAG, y= oldpeak_eq_st_depression )) + 
  geom_boxplot(fill="#2b8cbe",notch=TRUE) +
  labs(title="7d: oldpeak_eq_st_depression") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#Histogram & Density of age
c3<-ggplot(train, aes(x=age)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(age, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("age") +
  theme(plot.title = element_text(hjust = 0.5))

c4<-ggplot(train, aes(x=TARGET_FLAG, y= age )) + 
  geom_boxplot(fill="#3182bd",notch=TRUE) +
  labs(title="Distribution of age ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#Histogram & Density of max_heart_rate_achieved
c5<-ggplot(train, aes(x=max_heart_rate_achieved)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(max_heart_rate_achieved, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("max_heart_rate_achieved") +
  theme(plot.title = element_text(hjust = 0.5))

c6<-ggplot(train, aes(x=TARGET_FLAG, y= max_heart_rate_achieved )) + 
  geom_boxplot(fill="#bdc9e1",notch=TRUE) +
  labs(title="Distribution of max_heart_rate_achieved ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#Histogram & Density of max_predicted_heart_rate_achieved
c7<-ggplot(train, aes(x=max_predicted_heart_rate_achieved)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="#3182bd") +
  geom_density(alpha=.2, fill="#a6bddb") + # Overlay with transparent density plot
  geom_vline(aes(xintercept=mean(max_predicted_heart_rate_achieved, na.rm=T)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=1) +
  ggtitle("max_predicted_heart_rate_achieved") +
  theme(plot.title = element_text(hjust = 0.5))

c8 <-ggplot(train, aes(x=TARGET_FLAG, y= max_predicted_heart_rate_achieved)) + 
  geom_boxplot(fill="#045a8d",notch=TRUE) +
  labs(title="7e: max_predicted_heart_rate_achieved") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

multiplot(c1, c2, c3, c4, c5, c6, c7, c8, cols=4)

#Outlier Analysis
quantile(train$slope_of_peak_exercise_st_segment , c(.01, .05, .95, .99))
quantile(train$resting_blood_pressure , c(.01, .05, .95, .99))
quantile(train$chest_pain_type , c(.01, .05, .95, .99))
quantile(train$num_major_vessels , c(.01, .05, .95, .99))
quantile(train$resting_ekg_results , c(.01, .05, .95, .99))
quantile(train$serum_cholesterol_mg_per_dl , c(.01, .05, .95, .99))
quantile(train$oldpeak_eq_st_depression , c(.01, .05, .95, .99))
quantile(train$age , c(.01, .05, .95, .99))
quantile(train$max_heart_rate_achieved , c(.01, .05, .95, .99))
quantile(train$max_predicted_heart_rate_achieved , c(.01, .05, .95, .99))

#Correlation Matrix
subdatnumcor <- subset(train, select=c("slope_of_peak_exercise_st_segment","resting_blood_pressure","chest_pain_type","num_major_vessels",
                                       "resting_ekg_results","serum_cholesterol_mg_per_dl","oldpeak_eq_st_depression","age",
                                       "max_heart_rate_achieved","max_predicted_heart_rate_achieved"))
par(mfrow=c(1,1))  
corr <- round(cor(subdatnumcor),2)
ggcorrplot(corr, outline.col = "white", ggtheme = ggplot2::theme_gray, 
           colors = c("#E46726", "white", "#6D9EC1"),lab = TRUE)
par(mfrow=c(1,1))  

#Scatterplot Matrix 
require(lattice)
pairs(subdatnumcor, pch = 21)

GGally::ggpairs(as.data.frame(subdatnumcor))

###Scaterplots Colored by TARGET_FLAG###

sp1<-ggplot(train, aes(chest_pain_type, age, color = TARGET_FLAG)) + 
  geom_point(size=5, alpha=0.5) +
  scale_size_area() + 
  scale_color_brewer(palette="Set1") +
  ggtitle("Scatterplot of chest_pain_type vs. age") +
  theme_light()
sp1

sp2<-ggplot(train, aes(max_predicted_heart_rate_achieved, age, color = TARGET_FLAG)) + 
  geom_point(size=6, alpha=0.5) +
  scale_size_area() + 
  scale_color_brewer(palette="Set1") +
  ggtitle("Scatterplot of max_predicted_heart_rate_achieved vs. age") +
  theme_light()
sp2

sp3<-ggplot(train, aes(num_major_vessels, age, color = TARGET_FLAG)) + 
  geom_point(size=6, alpha=0.5) +
  scale_size_area() + 
  scale_color_brewer(palette="Set1") +
  ggtitle("Scatterplot of num_major_vessels vs. age") +
  theme_light()
sp3

multiplot(sp1, sp3, sp2, cols=1)

#sp1 + scale_color_manual(values=c("#999999", "#56B4E9"))
#sp2 + scale_color_manual(values=c("#999999", "#56B4E9"))

######EDA for Qualitative Variables#####

library(ggplot2)
#TARGET_FLAG
require(ggplot2)
ggplot(train) +
  geom_bar( aes(TARGET_FLAG) ) +
  ggtitle("TARGET_FLAG") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#thal 
require(ggplot2)
ggplot(train) +
  geom_bar( aes(thal) ) +
  ggtitle("thal") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#fasting_blood_sugar_gt_120_mg_per_dl 
require(ggplot2)
ggplot(train) +
  geom_bar( aes(fasting_blood_sugar_gt_120_mg_per_dl ) ) +
  ggtitle("fasting_blood_sugar_gt_120_mg_per_dl") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#sex  
require(ggplot2)
ggplot(train) +
  geom_bar( aes(sex  ) ) +
  ggtitle("sex") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#exercise_induced_angina  
require(ggplot2)
ggplot(train) +
  geom_bar( aes(exercise_induced_angina  ) ) +
  ggtitle("exercise_induced_angina") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#age_bin  
require(ggplot2)
ggplot(train) +
  geom_bar( aes(age_bin) ) +
  ggtitle("age_bin") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

###Crosstabs
attach(train)
library(gmodels)

#thal 
CrossTable(TARGET_FLAG,thal, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(train, aes(thal ,fill = TARGET_FLAG)) + geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#a6bddb", "#2b8cbe"))+
  theme_minimal()
l <- l + geom_histogram(stat="count")
tapply(as.numeric(train$TARGET_FLAG) - 1 , train$thal,mean)

A1 <- ggplot(train, aes(x = thal, fill = TARGET_FLAG)) + geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#a6bddb", "#2b8cbe"))+
  theme_minimal() + coord_flip()
grid.arrange(l,A1, nrow = 2, top = textGrob("8a: thal",gp=gpar(fontsize=15,font=2)))

#fasting_blood_sugar_gt_120_mg_per_dl 
CrossTable(TARGET_FLAG, fasting_blood_sugar_gt_120_mg_per_dl , prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(train, aes(fasting_blood_sugar_gt_120_mg_per_dl ,fill = TARGET_FLAG))+ geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#a6bddb", "#2b8cbe"))+
  theme_minimal()
l <- l + geom_histogram(stat="count")
tapply(as.numeric(train$TARGET_FLAG) - 1 , train$ fasting_blood_sugar_gt_120_mg_per_dl ,mean)

A <- ggplot(train, aes(x = fasting_blood_sugar_gt_120_mg_per_dl , fill = TARGET_FLAG)) + geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#a6bddb", "#2b8cbe"))+theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2,top = textGrob("fasting_blood_sugar_gt_120_mg_per_dl",gp=gpar(fontsize=15,font=2)))

#sex 
CrossTable(TARGET_FLAG, sex , prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(train, aes(sex ,fill = TARGET_FLAG))+ geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#9ecae1", "#3182bd"))+
  theme_minimal()
l <- l + geom_histogram(stat="count")
tapply(as.numeric(train$TARGET_FLAG) - 1 , train$ sex ,mean)

A2 <- ggplot(train, aes(x = sex , fill = TARGET_FLAG)) + geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#9ecae1", "#3182bd"))+theme_minimal() + coord_flip()
grid.arrange(l,A2, nrow = 2,top = textGrob("8b: sex",gp=gpar(fontsize=15,font=2)))

#exercise_induced_angina 
CrossTable(TARGET_FLAG, exercise_induced_angina , prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(train, aes(exercise_induced_angina ,fill = TARGET_FLAG))+ geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#9ecae1", "#08306b"))+
  theme_minimal()
l <- l + geom_histogram(stat="count")
tapply(as.numeric(train$TARGET_FLAG) - 1 , train$ exercise_induced_angina ,mean)

A3 <- ggplot(train, aes(x = exercise_induced_angina , fill = TARGET_FLAG)) + geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#9ecae1", "#08306b"))+theme_minimal() + coord_flip()
grid.arrange(l,A3, nrow = 2,top = textGrob("8c: exercise_induced_angina ",gp=gpar(fontsize=15,font=2)))

#age_bin
CrossTable(TARGET_FLAG, age_bin , prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(train, aes(age_bin ,fill = TARGET_FLAG))+ geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#c6dbef", "#08519c"))+
  theme_minimal()
l <- l + geom_histogram(stat="count")
tapply(as.numeric(train$TARGET_FLAG) - 1 , train$ age_bin ,mean)

A4 <- ggplot(train, aes(x = age_bin , fill = TARGET_FLAG)) + geom_bar(position = 'fill')+
  scale_fill_manual(values=c("#c6dbef", "#08519c"))+theme_minimal() + coord_flip()
grid.arrange(l,A4, nrow = 2,top = textGrob("8d: age_bin ",gp=gpar(fontsize=15,font=2)))

multiplot(A1, A2, A3, A4, cols=2)

####################################### Set up Data for Analysis ######################################

#Standardize
#Training Dataset
preObj <- preProcess(train[,-c(1,3,7,11,14,15,16)], method=c("center", "scale"))
x.train <- predict(preObj, train)
str(x.train)
summary(x.train)

x.train$patient_id <- NULL  #Remove id column
#x.train$x.variable <- NULL  #Remove x.variable example purposes

#Validation Dataset
preObj <- preProcess(validation[,-c(1,3,7,11,14,15,16)], method=c("center", "scale"))
x.validation <- predict(preObj, validation)
str(x.validation)
summary(x.validation)

x.validation$patient_id <- NULL  #Remove patient_id column
#x.validation$x.variable <- NULL  #Remove x.variable example purposes

#Test Dataset
preObj <- preProcess(test[,-c(1,3,7,11,14,15,16)], method=c("center", "scale"))
x.test <- predict(preObj, test)
str(x.test)
summary(x.test) 

####################################### Build Models ######################################

####Logistic Regression Model###
#Full Model for Variable Selection & Baseline
model.logfull <- glm(TARGET_FLAG ~ ., x.train, family=binomial("logit"))
varImp(model.logfull)

#Stepwise Regression for Variable Selection
model.lower = glm(TARGET_FLAG ~ 1, x.train, family = binomial(link="logit"))
model.logfull <- glm(TARGET_FLAG ~ ., x.train, family=binomial("logit"))
stepAIC(model.lower, scope = list(upper=model.logfull), direction="both", test="Chisq", data=x.train)

model.log2 <- glm(TARGET_FLAG ~ chest_pain_type + thal + num_major_vessels + 
                    oldpeak_eq_st_depression + sex, x.train, family=binomial("logit")) 

summary(model.log2) 
Anova(model.log2, type="II", test="Wald")
varImp(model.log2)
nagelkerke(model.log2)
vif(model.log2)

#Performance Metrics
AIC(model.log2) 
BIC(model.log2) 

##Performance on Validation Set
glm.pred <- predict(model.log2, x.validation, type="response") 

#Confusion Matrix
glm.pred  <- ifelse(glm.pred  > 0.5,1,0)
xtab.log1=table(glm.pred, x.validation$TARGET_FLAG)
confusionMatrix(xtab.log1, positive = "1") #0.8222

#ROC Curve
library(ROCR)
prob <- predict(model.log2, newdata=x.validation, type="response")
pred <- prediction(prob, x.validation$TARGET_FLAG)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf,lwd=2, col="#3182bd", main="ROC Curve", colorize=TRUE)
abline(a=0, b=1)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.80725

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(model.log2, newdata = x.validation, type = "response")

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #N/A

###LDA###
model.lda2 <- lda(TARGET_FLAG ~ chest_pain_type + thal + num_major_vessels + 
                    oldpeak_eq_st_depression + sex, 
                  x.train, family=binomial("logit"))

##Performance on Validation Set
lda.pred <- predict(model.lda2, x.validation)$posterior[,2] 

#Confusion Matrix
lda.pred  <- ifelse(lda.pred  > 0.5,1,0)
xtab.lda1=table(lda.pred, x.validation$TARGET_FLAG)
confusionMatrix(xtab.lda1, positive = "1") #0.8222

#ROC Curve
library(ROCR)

test <-  predict(model.lda2, x.validation)$posterior
pred <- prediction(test[,2], x.validation$TARGET_FLAG)
perf <- performance(pred, "tpr", "fpr")
plot(perf,lwd=2, col="#3182bd", main="ROC Curve", colorize=TRUE)
abline(a=0, b=1)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.82075

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(model.lda2, x.validation)$posterior[,2] 

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #0.5954447

###FDA Model###
library(mda)
detach(package:ModelMetrics)
set.seed(1)
fda.model=fda(TARGET_FLAG~chest_pain_type + thal + num_major_vessels + 
                oldpeak_eq_st_depression + sex,data= x.train) 
fda.model

##Performance on Validation Set
prediction_fda<- predict(fda.model, x.validation, type="class")

#Confusion Matrix
library(caret)

xtab.fda=table(prediction_fda, x.validation$TARGET_FLAG)
xtab.fda
confusionMatrix(prediction_fda,x.validation$TARGET_FLAG, positive = "1")#0.8222

#AUC
library(ModelMetrics)
prob <- predict(fda.model, newdata=x.validation, type='posterior')
auc<-auc(x.validation$TARGET_FLAG,prob[,2])
auc #0.82075

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(fda.model, newdata = x.validation, type = "posterior")[,2]

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #0.603699

### KNN ###

detach(package:ModelMetrics)

#Find K using CV Methods manual grid search
#set.seed(7)
#library(caret)
#control<- trainControl(method="cv", number=5)
#grid <- expand.grid(k=c(1:25))
#model <- train(TARGET_FLAG~., data=x.train_boost, method="knn", trControl=control, tuneGrid=grid,preProcess = c("center","scale"))
#plot(model) #k=21

#Find K using CV Methods automatic grid search
set.seed(400)
ctrl<- trainControl(method="repeatedcv",repeats = 5) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(TARGET_FLAG ~ ., data = x.train, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
knnFit
plot(knnFit) #k=21

#Confusion Matrix
KNN.pred <- predict(knnFit,x.validation)
confusionMatrix(KNN.pred, x.validation$TARGET_FLAG) #0.8444   

#AUC
library(ROCR)
KNN.pred <- predict(knnFit,x.validation, type = "prob")
pred_val <-prediction(KNN.pred[,2],x.validation$TARGET_FLAG)
perf_auc <- performance(pred_val,"auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.8905

#ROC Curve
perf_auc <- performance(pred_val, "tpr", "fpr")
plot(perf_auc, col = "#3182bd", lwd = 3)

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(knnFit,x.validation,type="prob")[,2]
str(x.validation)

actual <- x.validation[c("TARGET_FLAG")]
x.validation

actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #0.3882528

##### Decision Tree #####
detach(package:ModelMetrics)
#Fit Full Decision Tree
model.treefull<- tree(TARGET_FLAG ~ ., x.train)
summary(model.treefull)
model.treefull

#Plot Full Tree
plot(model.treefull)
text(model.treefull,pretty=0)

# Evaluate Full Tree on Validation Set
post.valid.treefull <- predict(model.treefull, x.validation, type="class") 

##Performance for Full Tree on Validation Set
table(post.valid.treefull, x.validation$TARGET_FLAG) # classification table
mean(post.valid.treefull== x.validation$TARGET_FLAG) #Accuracy 0.7222
treeError <- mean(post.valid.treefull!= x.validation$TARGET_FLAG) #Error 0.2777778
treeError
xtab.treefull=table(post.valid.treefull, x.validation$TARGET_FLAG)
confusionMatrix(xtab.treefull, positive = "1")

### Use Cross-Validation to Prune Tree ###
set.seed(3)
cv.treeprune=cv.tree(model.treefull,FUN=prune.misclass)
names(cv.treeprune)
cv.treeprune

#We plot the error rate as a function of both size and k.
par(mfrow=c(1,2))
plot(cv.treeprune$size, cv.treeprune$dev,type="b")
plot(cv.treeprune$k, cv.treeprune$dev,type="b")
par(mfrow=c(1,1))

#Apply Pruned Tree
model.treeprune=prune.misclass(model.treefull,best=2)
plot(model.treeprune)
text(model.treeprune,pretty=0)

##Evaluate Pruned Tree on Validation Set
post.valid.treeprune <- predict(model.treeprune, x.validation, type="class") 

#Performance for Pruned Tree on Validation Set
table(post.valid.treeprune, x.validation$TARGET_FLAG) # classification table
mean(post.valid.treeprune== x.validation$TARGET_FLAG) #Accuracy 0.7111
treeError <- mean(post.valid.treeprune!= x.validation$TARGET_FLAG) #Error 0.2888889
treeError
xtab.treeprune=table(post.valid.treeprune, x.validation$TARGET_FLAG)
confusionMatrix(xtab.treeprune, positive = "1")

#AUC
library(ModelMetrics)
prob <- predict(model.treeprune, newdata=x.validation, type='vector')
auc<-auc(x.validation$TARGET_FLAG,prob[,2])
auc #0.7125

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(model.treeprune, newdata = x.validation, type = "vector")[,2]

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #0.5583778

###Bagging###
detach(package:ModelMetrics)
library(randomForest)
x.validation$P_TARGET_FLAG <- NULL  #Remove 
set.seed(1)
bag.model=randomForest(TARGET_FLAG~.-age_bin,data= x.train, mtry=14, importance=TRUE,ntree=1050)
bag.model
##Performance on Validation Set
prediction_bag<- predict(bag.model, x.validation, type="class")

#Confusion Matrix
library(caret)
xtab.bag=table(prediction_bag, x.validation$TARGET_FLAG)
xtab.bag 
confusionMatrix(prediction_bag,x.validation$TARGET_FLAG, positive = "1") #0.7444

#AUC
library(ModelMetrics)
prob <- predict(bag.model, newdata=x.validation, type='prob')
auc<-auc(x.validation$TARGET_FLAG,prob[,2])
auc #0.8455 

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(bag.model, newdata = x.validation, type = "prob")[,2]

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #0.4310598 

###Random Forest Model###
detach(package:ModelMetrics)
library(caret)

#Random Search for mtry
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(1)
metric <- "Accuracy"
mtry <- sqrt(ncol(x.train))
rf_random <- train(TARGET_FLAG~chest_pain_type + thal + num_major_vessels + 
                     oldpeak_eq_st_depression + sex + 
                     max_predicted_heart_rate_achieved +serum_cholesterol_mg_per_dl, data=x.train, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random) #mtry=1

#Grid Search for mtry
#control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
#set.seed(1)
#metric <- "Accuracy"
#tunegrid <- expand.grid(.mtry=c(1:15))
#rf_gridsearch <- train(TARGET_FLAG~chest_pain_type + thal + num_major_vessels + 
                         #oldpeak_eq_st_depression + sex + 
                         #max_predicted_heart_rate_achieved +serum_cholesterol_mg_per_dl, data=x.train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
#print(rf_gridsearch)
#plot(rf_gridsearch) #mtry=1

#Manual Search for ntrees
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(x.train))))
modellist <- list()
for (ntree in c(500, 1000, 1500, 2000, 2500)) {
  set.seed(1)
  fit <- train(TARGET_FLAG~chest_pain_type + thal + num_major_vessels + 
                 oldpeak_eq_st_depression + sex + 
                 max_predicted_heart_rate_achieved +serum_cholesterol_mg_per_dl, data=x.train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit 
}

results <- resamples(modellist)
summary(results)
dotplot(results) #ntree = 2000

#Custom Grid Search
#customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
#customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
#customRF$grid <- function(x, y, len = NULL, search = "grid") {}
#customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  #randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
#}
#customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  #predict(modelFit, newdata)
#customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
#  predict(modelFit, newdata, type = "prob")
#customRF$sort <- function(x) x[order(x[,1]),]
#customRF$levels <- function(x) x$classes

#control <- trainControl(method="repeatedcv", number=10, repeats=3)
#tunegrid <- expand.grid(.mtry=c(1:15), .ntree=c(500, 1000, 1500, 2000, 2500))
#set.seed(1)
#custom <- train(TARGET_FLAG~chest_pain_type + thal + num_major_vessels + 
 #                 oldpeak_eq_st_depression + sex + 
  #                max_predicted_heart_rate_achieved +serum_cholesterol_mg_per_dl, data=x.train, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
#summary(custom)
#plot(custom) #mtry=1 & ntrees = 2000

#Variable Importance
set.seed(1)
control=trainControl((method="repeatedcv"),number=10,repeats=5)
control
model=train(TARGET_FLAG~chest_pain_type + thal + num_major_vessels + 
              oldpeak_eq_st_depression + sex +
              max_predicted_heart_rate_achieved +serum_cholesterol_mg_per_dl,data=x.train,method="rpart",preProcess="scale",trControl=control,tuneLength=5)
model
importance=varImp(model,scale=FALSE)
importance

library(randomForest)
detach(package:ModelMetrics)
set.seed(1)
rf.model=randomForest(TARGET_FLAG~chest_pain_type + thal + num_major_vessels + 
                        oldpeak_eq_st_depression + sex + 
                        max_predicted_heart_rate_achieved +serum_cholesterol_mg_per_dl,data= x.train,importance=TRUE,mtry=1, ntree=2000) #2150

rf.model
varImpPlot(rf.model)

##Performance on Validation Set
prediction_rf<- predict(rf.model, x.validation, type="class")

#Confusion Matrix
library(caret)
xtab.rf=table(prediction_rf, x.validation$TARGET_FLAG)
xtab.rf #0.8333
confusionMatrix(prediction_rf,x.validation$TARGET_FLAG, positive = "1") #0.8333

#AUC
library(ModelMetrics)
prob <- predict(rf.model, newdata=x.validation, type='prob')
auc<-auc(x.validation$TARGET_FLAG,prob[,2])
auc #0.8975 

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(rf.model, newdata = x.validation, type = "prob")[,2]

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #0.3655469 

### Gradient Boosting Machines ###
detach(package:ModelMetrics)
library(gbm)

x.train.boost= x.train
x.train.boost$TARGET_FLAG=ifelse(x.train.boost$TARGET_FLAG == "1",1,0)

#Find ideal parameter using CV using manual grid search
#set.seed(7)
#control <- trainControl(method="cv", number=5)
#grid <- expand.grid(interaction.depth=seq(1,7,by=2),n.trees=seq(50,1500,by=50),shrinkage=c(0.01,0.1),n.minobsinnode = 10)
#model <- train(TARGET_FLAG~., x.train.boost, method="gbm", trControl=control, tuneGrid=grid)
#print(model)
#The final values used for the model were n.trees = 50, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.


#Find ideal parameter using CV using automatic grid search
#library(caret)
#fitControl <- trainControl(method="repeatedcv", number=5, repeats=5)
#set.seed(1)
#gbmFit <- train(TARGET_FLAG ~ .-age_bin, data = x.train.boost,
           #     method = "gbm", trControl = fitControl, verbose = FALSE, tuneLength=5)
#gbmFit

#The final values used for the model were n.trees = 100, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

#Fit Boost
set.seed(1)
boost.model<-gbm(TARGET_FLAG~.-age_bin,data = x.train.boost,distribution="bernoulli",n.trees=50,shrinkage = 0.1,interaction.depth=1, n.minobsinnode = 10)

summary(boost.model) #chest_pain_type, thal, num_major_vessels, oldpeak_eq_st_depression, sex

#Evaluate on Validation Set
boost.pred<- predict(boost.model, x.validation, type="response", n.trees = 50) # n.valid 

#Confusion Matrix
boost.pred  <- ifelse(boost.pred  > 0.50,1,0)
xtab.boost1=table(boost.pred, x.validation$TARGET_FLAG)
confusionMatrix(xtab.boost1, positive = "1") #0.8556

#AUC
library(ROCR)
library(cvAUC)
labels <- x.validation[,"TARGET_FLAG"]
AUC(predictions = boost.pred, labels = labels) #0.8475

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(boost.model, newdata = x.validation, type = "response", n.trees = 50)

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #0.3698292 

###Neural Network###
library(caret)
#Use CV to find ideal parameters
fitControl <- trainControl(method="cv", number=5)
set.seed(1)

nnetFit <- train(TARGET_FLAG ~chest_pain_type + thal + num_major_vessels + 
                   oldpeak_eq_st_depression + sex, x.train,
                 method = "nnet", trControl = fitControl, verbose = FALSE)
nnetFit
plot(nnetFit)

#The final values used for the model were size = 5 and decay = 0.1.

set.seed(800)
model.nnet1 <- nnet(TARGET_FLAG ~chest_pain_type + thal + num_major_vessels + 
                      oldpeak_eq_st_depression + sex, x.train, size = 5, decay=0.1, maxit=1000)
model.nnet1 

#Performance on Validation Set
nnet.pred <- predict(model.nnet1, newdata=x.validation,type="class")

#Confusion Matrix
#table(nnet.pred,x.validation$TARGET_FLAG)
#(46+29)/90 #0.8333

nn.table=table(nnet.pred,x.validation$TARGET_FLAG)
confusionMatrix(nn.table,positive = "1") #0.8333

#ROC Curve
library(ROCR)

prob <- predict(model.nnet1, newdata=x.validation, type="raw")
library(ROCR)
pred = prediction(prob, x.validation$TARGET_FLAG)
perf = performance(pred, "tpr", "fpr")
plot(perf,lwd=2, col="#3182bd", main="ROC Curve", colorize=TRUE)
abline(a=0, b=1)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.87425

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(model.nnet1, newdata = x.validation, type = "raw")

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #0.4112035

### xgboost ###
detach(package:ModelMetrics)

x.train_boost= x.train
x.validation_boost= x.validation
x.test_boost= x.test

x.train_boost$TARGET_FLAG<- factor(ifelse(x.train_boost$TARGET_FLAG==0, "No", "Yes"))
x.validation_boost$TARGET_FLAG<- factor(ifelse(x.validation_boost$TARGET_FLAG==0, "No", "Yes"))
x.test_boost$TARGET_FLAG<- factor(ifelse(x.test_boost$TARGET_FLAG==0, "No", "Yes"))

x.train_boost$fasting_blood_sugar_gt_120_mg_per_dl <- as.numeric(x.train_boost$fasting_blood_sugar_gt_120_mg_per_dl)
x.train_boost$sex  <- as.numeric(x.train_boost$sex)
x.train_boost$exercise_induced_angina  <- as.numeric(x.train_boost$exercise_induced_angina )
x.train_boost$thal <- as.numeric(x.train_boost$thal)
x.train_boost$age_bin <- as.numeric(x.train_boost$age_bin)

x.validation_boost$fasting_blood_sugar_gt_120_mg_per_dl <- as.numeric(x.validation_boost$fasting_blood_sugar_gt_120_mg_per_dl)
x.validation_boost$sex  <- as.numeric(x.validation_boost$sex)
x.validation_boost$exercise_induced_angina  <- as.numeric(x.validation_boost$exercise_induced_angina )
x.validation_boost$thal <- as.numeric(x.validation_boost$thal)
x.validation_boost$age_bin <- as.numeric(x.validation_boost$age_bin)

x.test_boost$fasting_blood_sugar_gt_120_mg_per_dl <- as.numeric(x.test_boost$fasting_blood_sugar_gt_120_mg_per_dl)
x.test_boost$sex  <- as.numeric(x.test_boost$sex)
x.test_boost$exercise_induced_angina  <- as.numeric(x.test_boost$exercise_induced_angina )
x.test_boost$thal <- as.numeric(x.test_boost$thal)
x.test_boost$age_bin <- as.numeric(x.test_boost$age_bin)

x.validation_boost$P_TARGET_FLAG <- NULL  #Remove 

str(x.train_boost)
str(x.validation_boost)
str(x.test_boost)

set.seed(1)
ControlParameters <- trainControl(method = "repeatedcv",
                                  number = 5,
                                  savePredictions = TRUE,
                                  classProbs = TRUE)

#Manual
#xgb_grid_1 = expand.grid(
 # nrounds = 50, 
  #max_depth = c(1,2,3), 
  #eta = c(0.05,0.1,0.2,0.3,0.4, 0.7), 
  #gamma = 0, 
  #colsample_bytree = c(0.3, 0.4,0.5, 0.6),
  #min_child_weight = c(0.2,0.3,0.4,0.5,0.6, 0.7),subsample =c(0.5, 1))


#The final values used for the model were nrounds = 50, max_depth = 1, eta = 0.1, gamma = 0, colsample_bytree = 0.4, min_child_weight = 0.6
#and subsample = 0.5.

#set.seed(300)
#modelxgboost <- train(TARGET_FLAG~chest_pain_type + thal + num_major_vessels + 
 #                       oldpeak_eq_st_depression + sex + serum_cholesterol_mg_per_dl+ max_predicted_heart_rate_achieved, 
  #                    data = x.train_boost,
   #                   method = "xgbTree",
    #                  trControl = ControlParameters,tuneGrid = xgb_grid_1)
#modelxgboost 

#Automatic
set.seed(1)
modelxgboost <- train(TARGET_FLAG~chest_pain_type + thal + num_major_vessels + 
                        oldpeak_eq_st_depression + sex + serum_cholesterol_mg_per_dl 
                      + max_predicted_heart_rate_achieved, 
                      data = x.train_boost,
                      method = "xgbTree",
                      trControl = ControlParameters)


#The final values used for the model were nrounds = 50, max_depth = 1, eta = 0.3, gamma = 0, colsample_bytree = 0.6, min_child_weight = 1
#and subsample = 0.5. 

modelxgboost
importance=varImp(modelxgboost,scale=FALSE)
importance

#Confusion Matrix
xgboost.pred<-predict(modelxgboost,x.validation_boost)
xgboost.table=table(xgboost.pred,x.validation_boost$TARGET_FLAG)
xgboost.table
confusionMatrix(xgboost.table,positive = "Yes") #0.8556

#AUC
library(ModelMetrics)
prob <- predict(modelxgboost,x.validation_boost,type="prob")
auc<-auc(x.validation_boost$TARGET_FLAG,prob[,2])
auc #0.89125

#LogLoss
set.seed(1)
x.validation_boost$P_TARGET_FLAG<- predict(modelxgboost,x.validation_boost,type="prob")[,2]

actual <- x.validation_boost[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation_boost[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
LogLoss/100 #0.370117

#########################################Predictions for Test Dataset & Final Submission########################################
set.seed(1)
#x.test$P_TARGET_FLAG<- predict(model.nnet1, newdata = x.test, type = "raw")#Nnet
#x.test$P_TARGET_FLAG<- predict(rf.model, newdata = x.test, type = "prob")[,2] #RF
x.test_boost$P_TARGET_FLAG<- predict(modelxgboost,x.test_boost,type="prob")[,2] #xgboost

#x.test$P_TARGET_FLAG<- predict(boost.model, newdata = x.test, type = "response", n.trees = 50) #gbm
#x.test$P_TARGET_FLAG<- predict(naiveBayes.model, newdata = x.test, type = "raw")[,2] #NaiveBayes
#x.test$P_TARGET_FLAG<- predict(model.lda2, x.test)$posterior[,2] #LDA
#x.test$P_TARGET_FLAG<- predict(model.qda2, x.test)$posterior[,2] #QDA
#x.test$P_TARGET_FLAG<- predict(model.log2, newdata = x.test, type = "response") #Logistic

#subset of data set for the deliverable "Scored data file"
#scores <- x.test[c("patient_id","P_TARGET_FLAG")]
scores <- x.test_boost[c("patient_id","P_TARGET_FLAG")] #xgboost

#Submission format file
submission_format <- read.csv("submission_format.csv", check.names=FALSE)
colnames(scores) <- colnames(submission_format)
write.csv(scores, file="submission3.csv", row.names=FALSE )

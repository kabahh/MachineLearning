Course Project: Machine Learning
==================================================================

### Executive Summary

Human Activity Recognition is an emerging research area in the last years and is gaining increasing attention by the computing research community. Devices such as Jawbone Up, Nike FuelBand, and Fitbit 
now make it possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement, enthusiasts who take 
measurements about themselves regularly to improve their health, find patterns in their behavior, and other reasons. Six participants were asked to perform barbell lifts correctly and incorrectly in 5 
different ways. The activities was scored from A to E:

	1. Exactly according to the specification (Class A)
	2. Throwing the elbows to the front (Class B)
	3. Lifting the dumbbell only halfway (Class C)
	4. Lowering the dumbbell only halfway (Class D)
	5. Throwing the hips to the front (Class E)

Using data from accelerometers on the belt, forearm, arm, and dumbell of participants, the objective of the analysis was to predict how well a participant did the activity. Utilizing the method Random 
Forest and a finalized version of the accelerometer data, we were able to predict the class of the activity with 98.3% accuracy. All analyses were conducted in R 3.0.2.

### Analysis Plan    

First, the necessary packages and data for analysis were loaded into R.

```{r}
setwd(getwd())
library(caret)
library(randomForest)
library(knitr)

train <- read.csv("pml-training.csv", na.strings=c("", "NA"))
test <- read.csv("pml-testing.csv", na.strings=c("", "NA"))
```         

Once the data was read in, basic descriptive statistics were run on the variables. The same amount of missingness was found in 100 variables of the 160 avaliable in the data (NA = 19,216). Using this 
information, a short function was written to connect the amount of missinginess in a variable to its corresponding column name. A smaller dataset (train_sml) was created to only include variables that 
were not missing 19,216 observations. To ensure that the correct variables were pulled into the smaller dataset, the function "identical" was used to compare the names of the variables that had no 
missingness to the variables in the smaller dataset.     

```{r}
summary(train[,146:149])
names <- as.matrix(names(train))
temp1 <- as.matrix(dim(train)[2],1)
for(i in 1:dim(train)[2]){
temp1[i] <- sum(is.na(train[,i]))
}
out <- cbind(names, temp1)
train_sml <- train[,-c(which(out[,2] == "19216"))]
out2 <- subset(out, out[,2] == "0")
identical(out2[,1], names(train_sml))
```   
     
The next step was to identify which variables were correlated with outcome "Classe". In order to use the "cor" function, the outcome variable had to be recoded as a numeric variable. When completed, the
recoding was verified via the "table" command.

```{r}
train_sml[,61] <- 1
train_sml[,61][train_sml[,60] == "A"] <- 1
train_sml[,61][train_sml[,60] == "B"] <- 2
train_sml[,61][train_sml[,60] == "C"] <- 3
train_sml[,61][train_sml[,60] == "D"] <- 4
train_sml[,61][train_sml[,60] == "E"] <- 5
table(train_sml[,60], train_sml[,61])
```          
The first seven variables corresponded to the users of the device and time stamping. These were not included in the correlation analysis nor in any analysis as the relationahip between these variables
and the outcome was not clearly justifiable or logical. The correlation analysis began at variable "roll_belt" to "magnet_forearm_z" from the small dataset (train_sml). A histogram of the absolute 
values of the correlations between "classe" to all other variables was created to investigate and identify how many variales were highly correlated. In the plot below, it can be seen that the majority
of the variables had a correlation with "classe" that ranged from absolute 0 to 0.10. Variables with this level of correlation were not of interest to the analysis and were excluded.    

```{r}
hist(abs(cor(train_sml[,c(8:59,61)])[53,]), 10, main="Histogram of Correlations with Numeric Classe", 
ylim=c(0,40),xlab="Absolute Value of Correlations")
text(0.8,5,"Note: Outlier is the correlation\nof Classe to itself")
abline(v=0.10, col="red", lwd = 2)
```         

Including only variables with acceptable levels of correlation (>=.10), the final analysis dataset contains 15 explanatory variables, 16 total in the dataset including the outcome (train_sml2).   

```{r}
temp3 <- as.numeric(which(abs(as.vector(cor(train_sml[,c(8:59,61)])[53,]))>= 0.10)) + 7
train_sml2 <- train_sml[,c(temp3)]
names(train_sml2)
```    
The final analysis dataset was partitioned into a secondary training set (training) and test set (testing). The training set contains 60% of the original train data, while the testing set contains the
remaining observations.

```{r}
trainIndex = createDataPartition(train_sml2[,16],p=0.6,list=FALSE)
training <- train_sml2[trainIndex,]
testing <- train_sml2[-trainIndex,]
```    
The method chosen to predict "classe" was Random Forest as it is highly accurate, can handle numerous explanatory variables, and can handle multilevel character outcomes. Instead of using the "train" 
function in the CARET package, the function "randomForest" was used to save computation time. The results between the two programs are comparable.

```{r RandomForestModel, cache=TRUE}
modFit.rf <- randomForest(training[,-16], training[,16], prox=TRUE)
``` 

Below are the results of the Random Forest modeling. From this, the estimated Out-of-Bag (OOB) error rate is 4.44%. This can be interpreted as, given new data, we expect that all but ~4.4% will be 
classified correctly. Next, the model was used to predict the "classe" outcome for the testing dataset. Looking at the results from the testing data prediction, the accuracy was 98.3%. This fall inline
with the expected error rate. This cross-validation of the model (training set predictions vs. testing set predictions) adds confidence to the accuracy of the model. 

```{r}
modFit.rf
pred <- predict(modFit.rf, testing)
confusionMatrix(pred,testing$classe)
```    
Lastly, the model was used to predict the outcomes for the original test data (test). The test data was reduced to only include the variables used in the prediction model (test2). Below is a sample of 
the predicted outcomes for the given variable values.

```{r}
vars <- names(training[,-16])
test2 <- test[,c(vars)]
head(cbind(as.character(predict(modFit.rf, test2)), test[,160]))
````

### Conclusion

The predicted results for the 20 test cases were submitted and were correct. This verified that the random forest approach was able to correctly classify each of the activities to the correct "classe"
with high accuracy.

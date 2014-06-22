##########################################
# Assignment 1 Coursea Practical Machine learning course
##########################################
#
# By ORatWork
#
# Created 		: 2014 06 08
# Last revision 	: 2014 06 08 
#
#
##########################################

rm(list=ls()) #will remove ALL objects
require(caret)
require(R2HTML)

setwd("D:/P R O J E C T S/Business Analytics/Coursera/Practical Machine Learning/Assignment 1")

HTMLStart(outdir="D:/P R O J E C T S/Business Analytics/Coursera/Practical Machine Learning/Assignment 1", file="PML_assignment",
  	 extension="html", echo=TRUE, HTMLframe=TRUE)

HTML.title("Practical Machine Learning Assignment", HR=1)

set.seed(12321) # init seed, to be able to replicate the analysis

##########################################
# write predictions to seperate files
#########################################

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


## Init

# read the trainig data set

HTML.title("Review and adjust available training data", HR=3)

RawData<-read.csv(file.choose(), 
			header=TRUE, sep=",", na.strings=c("","NA", "?"))

dim(RawData)
str(RawData)
summary(RawData)

HTML.title("Preprocess data and Exploratory data analysis", HR=3)

# work on a copy of the RawData set

df<-RawData

# reduce the data set to those columns which are at least 80% filled 

Threshold<-0.8 # 80% of filled rows per column

RemovedCols<-0
NbrRows<-dim(df)[1]

for (i in names(df)){
	tmp<-sum( !is.na( df[[i]]) )
	tmp<-tmp/NbrRows
	if (tmp<Threshold){
		df[[i]]<-NULL   # remove the column
		RemovedCols<-RemovedCols+1
 	}
}
print(RemovedCols)

# split into training and testing set
# data set is of moderate size, use a 60/40 split for Train and Test set

inTrain<-createDataPartition(df$classe, p=0.6, list=FALSE)
Train<-df[inTrain,]
Test<-df[-inTrain,]

# check for Near Zero Variance in Training Set

nsv<-nearZeroVar(Train)
print(nsv)

## remove near zero variance columns in both Train and Test
Train<-Train[,-nsv]
Test<-Test[,-nsv]

# make some feature plots
# get number of columns in the training set

Nbr<-dim(Train)[2]

# Set dimensions for feature plots to 6 
MaxDim<-6
c<-NULL
# create string of names to plot
for (j in 1:MaxDim) c<-append(c,names(Train)[j], after=length(c))
# create the feature plot

HTML.title("Feature plot of uncorrelated features", HR=2)
print(featurePlot(x=Train[,c], y=Train$classe, plot="pairs"))
HTMLplot() 

# based on EDA decided to remove X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp as potential predictors

RemoveCols<-c(1,2,3,4,5,6)

Train<-Train[,-RemoveCols]
Test<-Test[,-RemoveCols]

#cross validation of predictors

HTML.title("Cross validation of predictors", HR=3)

require(randomForest)

CVResult <- rfcv(Train[,-53], Train$classe, step=0.50)
HTML.title("Corss validation Plot", HR=2)
with(CVResult, plot(n.var, error.cv, log="x", type="o", lwd=2, ylab= "Error Rate", xlab="Number of predictors", main="Cross Validation Error rate vs Number of predictors"))
HTMLplot()

HTML.title("Train a classification model", HR=3)
 
# based on the outcome of rfcv it seems that between 13 - 52 vars are enough
# first fit a model using all 52 predictors 
# us cross validation method to train, use 5 as number of folds

trContr <- trainControl(method = "cv", number = 5)
modelFit<-train(classe~.,data=Train, method="rf", trControl=trContr ) 

HTML.title("Training results", HR=3)

modelFit
modelFit$finalModel

HTML.title("Variable importance plot", HR=2)
varImpPlot(modelFit$finalModel)
HTMLplot()

## get the 13 most important predictors

df.rfImportance <- data.frame(variable = names(modelFit$finalModel$importance[,1]), 
					importance = modelFit$finalModel$importance[,1])
df.rfImportance <- df.rfImportance[ order(-df.rfImportance[,2]),]

#convert from factor to character
df.rfImportance$variable<-as.character(df.rfImportance$variable)

#set threshold value for importance score to 14th element of ordered list. 
ThresholdImp<-df.rfImportance[14,]$importance

#Keep all columns with a higher importance level that Threshold

tmp<-df.rfImportance[df.rfImportance$importance>ThresholdImp,]$variable
NamesSmall<-append(tmp, c("classe"), after=length(tmp))

HTML.title("Variables in scope, based on cross validation plot, 13 in total", HR=2)
NamesSmall

# create small Train and Test set

TrainLess<-Train[,NamesSmall]
TestLess<-Test[,NamesSmall]

# fit the smaller model

SmallmodelFit<-train(classe~.,data=TrainLess, method="rf") 

HTML.title("Training results Smaller model", HR=3)

SmallmodelFit
SmallmodelFit$finalModel

HTML.title("Variable importance plot", HR=2)
varImpPlot(SmallmodelFit$finalModel)
HTMLplot()

## predictions with model and the smaller model

HTML.title("Test model Accuracy", HR=3)

Smallpredictions<-predict(SmallmodelFit, TestLess)
print(confusionMatrix(Smallpredictions, TestLess$classe))

## read in the test data for final prediction
##

TestData<-read.csv(file.choose(), 
			header=TRUE, sep=",", na.strings=c("","NA", "?"))

dim(TestData)
str(TestData)

## use SmallmodelFit for predictions 

HTML.title("Prediction", HR=3)

PredictTest<-as.character(predict(SmallmodelFit, TestData))
pml_write_files(PredictTest)
 
HTMLStop()




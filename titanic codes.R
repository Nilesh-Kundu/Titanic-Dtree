library(rpart)# tree models 
library(caret) # feature selection
library(rpart.plot) # plot dtree
library(ROCR) # model evaluation
library(e1071) # tuning model
library(RColorBrewer)
library(rattle)# optional, if you can't install it, it's okay
library(tree)
library(ISLR)

############################## 1 way of doing ########################################


setwd("C:\\Users\\ADMIN\\Desktop\\R Models\\Decision Tree")
Carseats <- read.csv("Titanic.csv")
head(Carseats)
tail(Carseats)
str(Carseats)
summary(Carseats)

Carseats <- Carseats[ -c(1,4,9,11) ]
## Let's also change the labels under the "status" from (0,1) to (normal, abnormal)   
Carseats$Pclass <- as.factor(Carseats$Pclass) 
Carseats$Survived <- factor(Carseats$Survived, levels = c(0, 1),labels = c('No', 'Yes'))

## Check the missing value (if any)
sapply(Carseats, function(x) sum(is.na(x)))

Carseats <- na.omit(Carseats)

## separating the independent and dependent variables
ind_var <- Carseats[,-1]
dep_var <- Carseats[,1]

## Now you can randomly split your data in to 70% training set and 30% test set   
set.seed(123)
train <- sample(1:nrow(Carseats), round(0.70*nrow(Carseats),0))
test <- -train
training <- Carseats[train,]
testing <- Carseats[test,]
test_Survived <- testing$Survived

# Decision Tree Model
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
fit <- rpart(Survived~., data = training, method = 'class', cp = 0.003)
rpart.plot(fit, extra = 106)

rattle()
fancyRpartPlot(fit, palettes = c("Greens", "Reds"), sub = "")
predictions <- predict(fit, testing, type="class")
mean(predictions != test_Survived)

conf.matrix <- table(testing$Survived, predictions)
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ":")
colnames(conf.matrix) <- paste("Predicted", colnames(conf.matrix), sep = ":")
print(conf.matrix)


############################
########Pruning#############
############################

printcp(fit)## check where xerror is lowest
bestcp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]

# Prune the tree using the best cp.
pruned <- prune(fit, cp = bestcp)
rpart.plot(pruned, extra = 106)

# Plot pruned tree
prp(pruned, faclen = 0, cex = 0.8, extra = 1)

rattle()
fancyRpartPlot(pruned, palettes = c("Greens", "Reds"), sub = "")

predictions <- predict(pruned, testing, type="class")
mean(predictions != testing$Survived)

predictions <- predict(pruned, ind_var, type="class")
confusionMatrix(data=predictions, reference=dep_var, positive="Yes")



# Advanced Plot
prp(pruned, main="Beautiful Tree",
    extra=106, 
    nn=TRUE, 
    fallen.leaves=TRUE, 
    branch=.5, 
    faclen=0, 
    trace=1, 
    shadow.col="gray", 
    branch.lty=3, 
    split.cex=1.2, 
    split.prefix="is ", 
    split.suffix="?", 
    split.box.col="lightgray", 
    split.border.col="darkgray", 
    split.round=.5)

write.csv(prediction,"bhul.csv")

#different way of getting confusion matrix and accuracy
prediction <-predict(pruned, testing, type = 'class')
table_mat <- table(testing$Survived, prediction)
table_mat
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))
round((32 + 18)/263,2)*100
100-(round((32 + 18)/263,2)*100)

evaluation <- function(fit, testing, "class") {
  cat("\nConfusion matrix:\n")
  prediction = predict(pruned, testing, type="class")
  xtab = table(prediction, testing$Survived)
  print(xtab)
  cat("\nEvaluation:\n\n")
  accuracy = sum(prediction == testing$Survived)/length(testing$Survived)
  precision = xtab[1,1]/sum(xtab[,1])
  recall = xtab[1,1]/sum(xtab[1,])
  f = 2 * (precision * recall) / (precision + recall)
  cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
  cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
  cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
  cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
}
evaluation(pruned, testing, "class")

conf.matrix <- table(testing$Survived, predictions)
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ":")
colnames(conf.matrix) <- paste("Predicted", colnames(conf.matrix), sep = ":")
print(conf.matrix)



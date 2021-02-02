# read in embedding
filepath = 'C:/Users/phill/Desktop/Stock2Vec/stock_risk_prediction/data'
setwd(filepath)
embeddings <- read.csv('S&P 500 Embeddings.csv')

# read in NPF and trim data set
data <- read.csv('2018_Financial_Data.csv')
net_profit_margin <- subset(data, select = c(Symbol,Net.Profit.Margin))

# combine data sets
df <- merge(embeddings, net_profit_margin, by="Symbol")

# divide data 30% test 70% train
sample_size <- floor(0.7*nrow(df))
train_index <- sample(seq_len(nrow(df)), size = sample_size)

# generate train and test data sets
train <- df[train_index,]
train <- train[complete.cases(train),]
test <- df[-train_index,]
test <- test[complete.cases(test),]

# train linear regression model
model <- lm(Net.Profit.Margin~Feature.1+Feature.2+Feature.3+Feature.4, data = train)

# predict NPM
prediction <- predict(model, newdata = test)
cor.test(prediction, test$Net.Profit.Margin, use = 'complete')

actual <- data.frame(cbind(actual=test$Net.Profit.Margin, predicted=prediction))
actual
cor(actual)

# diagnostic graphs
layout(matrix(c(1,2,3,4),2,2))
plot(model)

### REGULARIZATION ###

# remove NA samples
rtrain <- train[complete.cases(train),]

# cross validation: 5-fold CV x2
fitControl <- trainControl(method = 'repeatedcv', number = 5, repeats = 2)

# train model
glmFit <- train(Net.Profit.Margin~Feature.1+Feature.2+Feature.3+Feature.4, data = rtrain, method = 'penalized', trControl = fitControl)

# predict NPM
rtest <- test[complete.cases(test),]
rprediction <- predict(glmFit, newdata = rtest)
cor.test(rprediction, rtest$Net.Profit.Margin, use = 'complete')
#################################################################################
############## Lab Assignment II: Forecasting #########################
#################################################################################



library(MLTools)
library(fpp2)
library(ggplot2)
library(readr)
library(lmtest)
library(tseries)
library(TSA)
library(Hmisc)




## Load dataset ------------------------------------------------------------------------------------------
fdata <- read.table(file="DailyPrice_Load_Wind_Spain_2019_2020.csv", sep=";",header = TRUE,na.strings="NA")


#fdata <- read_excel("DailyPrice_Load_Wind_Spain_2019_2020.xlsx")
# fdata <- read.table("CarRegistrations.dat",header = TRUE)
head(fdata)

############SARIMA##########################



# Convert to time series object
y <- ts(fdata$Price, frequency = 7)
# for daily data
autoplot(y)




## Training and validation ------------------------------------------------------------
y.TR <- subset(y, end = length(y)-92) #Leave 3 month for validation
y.TV <- subset(y, start = length(y)-92+1)



## Identification and fitting frocess -------------------------------------------------------------------------------------------------------
autoplot(y.TR)



# Box-Cox transformation
Lambda <- BoxCox.lambda.plot(y.TR,7)
# Lambda <- BoxCox.lambda(y) #other option
z <- BoxCox(y.TR,Lambda)



# Differentiation: if the ACF decreases very slowly -> needs differenciation
ggtsdisplay(z,lag.max = 100)



# Regular Differentiation
Bz <- diff(z,differences = 1)
ggtsdisplay(Bz,lag.max = 100) #differences contains the order of differentiation



# Regular & Seasonal Differentiation
B12Bz <- diff(Bz, lag = 7, differences = 1)
ggtsdisplay(B12Bz,lag.max = 100)




# Fit seasonal model with estimated order
arima.fit <- Arima(y.TR,
                   order=c(1,1,2),
                   seasonal = list(order=c(0,1,1),period=7),
                   lambda = Lambda,
                   include.constant = FALSE)
summary(arima.fit) # summary of training errors and estimated coefficients
coeftest(arima.fit) # statistical significance of estimated coefficients
autoplot(arima.fit) # root plot

# Check residuals
CheckResiduals.ICAI(arima.fit, bins = 100, lag=60)
# If residuals are not white noise, change order of ARMA
ggtsdisplay(residuals(arima.fit),lag.max = 100)


arima.fit1 <- auto.arima(y.TR, trace=TRUE)
coeftest(arima.fit1) 
autoplot(arima.fit1)


# Check residuals
CheckResiduals.ICAI(arima.fit1, bins = 100, lag=60)
# If residuals are not white noise, change order of ARMA
ggtsdisplay(residuals(arima.fit1),lag.max = 100)




## Validation error for h = 1 -------------------------------------------------------------------------------------------------------
# Obtain the forecast in validation for horizon = 1 using the trained parameters of the model
y.TV.est <- y*NA
for (i in seq(length(y.TR)+1, length(y), 1)){# loop for validation period
  y.TV.est[i] <- forecast(subset(y,end=i-1), # y series up to sample i
                          model = arima.fit, # Model trained (Also valid for exponential smoothing models)
                          h=1)$mean # h is the forecast horizon
}



#Plot series and forecast
autoplot(y)+
  forecast::autolayer(y.TV.est)



#Compute validation errors
accuracy(y.TV.est,y)





###############TF#####################



#DEMAND+PRICE#



fdata_ts <- ts(fdata)
autoplot(fdata_ts, facets = TRUE)
#Seasonal time series
x <- ts(fdata$Demand,frequency = 7)/1
y <- ts(fdata$Price, frequency = 7)/1
ggtsdisplay(y, lag=100)



## Identification and fitting process -------------------------------------------------------------------------------------------------------



#### Fit initial FT model with large s
# This arima function belongs to the TSA package
TF.fit <- arima(y,
                order=c(1,1,0),
                seasonal = list(order=c(1,1,0),period=7),
                xtransf = x,
                transfer = list(c(0,15)), #List with (r,s) orders
                include.mean = TRUE,
                method="ML")



summary(TF.fit) # summary of training errors and estimated coefficients
coeftest(TF.fit) # statistical significance of estimated coefficients
# Check regression error to see the need of differentiation
TF.RegressionError.plot(y,x,TF.fit,lag.max = 100)
#NOTE: If this regression error is not stationary in variance,boxcox should be applied to input and output series.

TF.fit1 <- auto.arima(y,trace=TRUE)
summary(TF.fit1) # summary of training errors and estimated coefficients
coeftest(TF.fit1) # statistical significance of estimated coefficients
# Check regression error to see the need of differentiation
TF.RegressionError.plot(y,x,TF.fit1,lag.max = 100)
#NOTE: If this regression error is not stationary in variance,boxcox should be applied to input and output series.


TF.fit2 <- arima(y,
                 order=c(3,1,2),
                 seasonal = list(order=c(2,0,0),period=7),
                 xtransf = x,
                 transfer = list(c(0,15)), #List with (r,s) orders
                 include.mean = FALSE,
                 method="ML")

summary(TF.fit2) # summary of training errors and estimated coefficients
coeftest(TF.fit2) # statistical significance of estimated coefficients
# Check regression error to see the need of differentiation
TF.RegressionError.plot(y,x,TF.fit2,lag.max = 100)
#NOTE: If this regression error is not stationary in variance,boxcox should be applied to input and output series.



# Check numerator coefficients of explanatory variable
TF.Identification.plot(x,TF.fit)



#### Fit arima noise with selected
xlag = Lag(x,0) # b
xlag[is.na(xlag)]=0
arima.fit <- arima(y,
                   order=c(7,1,0),
                   seasonal = list(order=c(0,1,7),period=7),
                   xtransf = xlag,
                   transfer = list(c(0,0)), #List with (r,s) orders
                   include.mean = FALSE,
                   method="ML")
summary(arima.fit) # summary of training errors and estimated coefficients
coeftest(arima.fit) # statistical significance of estimated coefficients
# Check residuals
CheckResiduals.ICAI(arima.fit, lag=50)
# If residuals are not white noise, change order of ARMA
ggtsdisplay(residuals(arima.fit),lag.max = 50)



############### NON-LINEAR: MLP #####################

library(MLTools)
library(fpp2)
library(lmtest)
library(tseries) #contains adf.test function
library(TSA)
library(NeuralSens)
library(caret)
library(kernlab)
library(nnet)
library(NeuralNetTools)

#Load data
fdata <- read.table(file="DailyPrice_Load_Wind_Spain_2019_2020.csv", sep=";",header = TRUE,na.strings="NA")


#Initialize output and input variables
fdata.Reg <- fdata[,c(2,3,4)] 

View(fdata.Reg)

###Include lagged variables
#This can be done using the lag() function from the stats package but it works with time series objects
#The code is cleaner using the Lag() function from Hmisc package
library(Hmisc)
fdata.Reg$Dem_lag1 <- Lag(fdata$Demand,1)
fdata.Reg$Dem_lag7 <- Lag(fdata$Demand,7)
fdata.Reg$Wind_lag1 <- Lag(fdata$Wind,1)
fdata.Reg$Wind_lag7 <- Lag(fdata$Wind,7)
fdata.Reg$Price_lag1 <- Lag(fdata$Price,1)
fdata.Reg$Price_lag7 <- Lag(fdata$Price,7)


#Notice that the begining of the time series contains NA due to the new lagged series
head(fdata.Reg)

fdata.Reg.tr <- fdata.Reg
#Remove missing values
fdata.Reg.tr <- na.omit(fdata.Reg.tr)

## Initialize trainControl -----------------------------------------------------------------------
#Use resampling for measuring generalization error
#K-fold with 10 folds
ctrl_tune <- trainControl(method = "cv",                     
                          number = 10,
                          summaryFunction = defaultSummary,    #Performance summary for comparing models in hold-out samples.
                          savePredictions = TRUE)              #save predictions


#------------ Neural network --------------------------------
set.seed(150) #For replication
mlp.fit = train(form = Price~., #Use formula method to account for categorical variables
                data = fdata.Reg.tr, 
                method = "nnet",
                linout = TRUE,
                # tuneGrid = data.frame(size =5, decay = 0),
                tuneGrid = expand.grid(size = seq(5,15,length.out =3), decay =  10^(c(-3:0))),
                maxit = 200,
                preProcess = c("center","scale"),
                trControl = ctrl_tune, 
                metric = "RMSE")
mlp.fit #information about the resampling settings
ggplot(mlp.fit)+scale_x_log10()
plotnet(mlp.fit$finalModel) #Plot the network
SensAnalysisMLP(mlp.fit) #Statistical sensitivity analysis


#Predict training data
mlp_pred = predict(mlp.fit,  newdata = fdata.Reg.tr)  

#PlotModelDiagnosis(fdata.Reg.tr[,-1], fdata.Reg.tr[,1], mlp_pred, together = TRUE)

#Error measurements
accuracy(fdata.Reg.tr[,3],mlp_pred)

plot(fdata.Reg.tr[,3],type="l")
lines(mlp_pred,col = "red")



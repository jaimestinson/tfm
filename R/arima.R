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
library(forecast)
library(smooth)

## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Load dataset ------------------------------------------------------------------------------------------
fdata <- read.table(file="Demanda.csv", sep=";",header = TRUE,na.strings="NA")
fdata <- read.table(file="dataset.csv", sep=";",header = TRUE,na.strings="NA",nrows = 17544)



#fdata <- read_excel("DailyPrice_Load_Wind_Spain_2019_2020.xlsx")
# fdata <- read.table("CarRegistrations.dat",header = TRUE)
head(fdata)
tail(fdata)

############SARIMA##########################



# Convert to time series object
y <- ts(fdata$precio_spot, frequency = 24)
# for daily data
autoplot(y)


n <- length(y)


## Training and validation ------------------------------------------------------------
y.TR <- subset(y, end = 0.7*n) #Leave 30% for validation
y.TV <- subset(y, start = 0.7*n + 1)



## Identification and fitting process -------------------------------------------------------------------------------------------------------
autoplot(y.TR)



# Box-Cox transformation
Lambda <- BoxCox.lambda.plot(y.TR,350)
# Lambda <- BoxCox.lambda(y) #other option
z <- BoxCox(y.TR,Lambda)



# Differentiation: if the ACF decreases very slowly -> needs differenciation
ggtsdisplay(z,lag.max = )



# Regular Differentiation
Bz <- diff(z,differences = 1)
ggtsdisplay(Bz,lag.max = 100) #differences contains the order of differentiation



# Regular & Seasonal Differentiation
B12Bz <- diff(Bz, lag = 7, differences = 1)
ggtsdisplay(B12Bz,lag.max = 100)




# Fit seasonal model with estimated order
arima.fit <- Arima(y.TR,
                   order=c(8,1,1),
                   seasonal = list(order=c(1,1,2),period=24),
                   lambda = Lambda,
                   include.constant = FALSE)
summary(arima.fit) # summary of training errors and estimated coefficients
coeftest(arima.fit) # statistical significance of estimated coefficients
autoplot(arima.fit) # root plot

msarima.fit <- msarima(y.TR, orders=list(ar=c(2,2,1), i=c(1,1,1), ma=c(2,2,1)), lags=c(1,24,168), h=168, holdout=TRUE, ic=c("AIC","BIC"),loss=c("MAE"))



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

## Validation error for h = 1 -------------------------------------------------------------------------------------------------------
# Obtain the forecast in validation for horizon = 1 using the trained parameters of the model
y.TV.est1 <- y*NA
for (i in seq(length(y.TR)+1, length(y), 1)){# loop for validation period
  y.TV.est[i] <- forecast(subset(y,end=i-1), # y series up to sample i
                          model = arima.fit1, # Model trained (Also valid for exponential smoothing models)
                          h=1)$mean # h is the forecast horizon
}



#Plot series and forecast
autoplot(y)+
  forecast::autolayer(y.TV.est)



#Compute validation errors arimafit1 (2,1,4) (?)
accuracy(y.TV.est,y)


#Compute validation errors arimafit1 (1,1,2) (0,1,1)
accuracy(y.TV.est,y)

# Check fitted forecast
autoplot(y, series = "Real")+
  forecast::autolayer(arima.fit$fitted, series = "Fitted")


# Perform future forecast
y_est <- forecast(arima.fit, h=12)
autoplot(y_est)


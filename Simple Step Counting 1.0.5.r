#### R Script for Counting Steps ####
#'
#' Description:
#'
#'
#' |------------History of Changes--------------------|
#' |   Date   | Initials | Version | Description      |
#' |--------------------------------------------------|
#' | 23/04/20 |    CS    | 1.0.0   | Initial Creation |
#' | 23/04/20 |    CS    | 1.0.1   | Adding outputs   |
#' | 30/04/20 |    jl    | 1.0.1jl | optimised        |
#' | 01/05/20 |    jl    | 1.0.1jl2| hysteresis       |
#' | 05/05/20 |    CS    | 1.0.4   | Function Creation|
#' | 06/05/20 |    jl    | 1.0.4jl | Tidy & mod o/p   |
#' | 06/05/20 |    CS    | 1.0.5   | Function Update  |
#' |--------------------------------------------------|
#'
#'
#### Code ####

# Clear the decks
rm(list=ls())
gc()

# Libraries
library(GENEAread)
library(signal)

# Data to process
Datafile = "jl_right wrist_024600_2017-08-16 17-43-43_walk.bin"
Datafile = "Data/RunWalk.bin"
start = 0
end = 1
accdata = read.bin(Datafile, start = start, end = end)

#### Creating a function ####

#' Function to calculate the number and variance of the steps in the data.
#'
#' @title Step Counter
#' @param data The data to use for calculating the steps. This should either an AccData object or a vector.
#' @param samplefreq The sampling frequency of the data, in hertz,
#' when calculating the step number (default 100).
#' @param smlen single integer number of data points used as a window
#' when counting zero crossing events
#' @param filterorder single integer, order of the Chebyshev bandpass filter,
#' passed to argument n of \code{\link[signal]{cheby1}}.
#' @param boundaries length 2 numeric vector specifying lower and upper bounds
#' of Chebychev filter (default \code{c(0.5, 5)} Hz),
#' passed to argument W of \code{\link[signal]{butter}} or \code{\link[signal]{cheby1}}.
#' @param Rp the decibel level that the cheby filter takes, see \code{\link[signal]{cheby1}}.
#' @param plot.it single logical create plot of data and zero crossing points (default \code{FALSE}).
#' @param Centre If Centre set to true (default) then the step counter zeros the xz series before filtering.
#' @param fun character vector naming functions by which to summarize steps.
#' "count" is an internally implemented summarizing function that returns step count.
#' @param verbose single logical should additional progress reporting be printed at the console? (default FALSE).
#' @return Returns a vector with length fun.
#' @export
#' @importFrom signal cheby1
#' @examples
#' d1 <- sin(seq(0.1, 100, 0.1))/2 + rnorm(1000)/10 + 1
#' Steps4 = StepCounter(d1)
#' length(Steps4)
#' mean(Steps4)
#' sd(Steps4)
#' plot(Steps4)

StepCounter <- function(AccData, 
                        filterorder = 2,
                        Rp = 3,
                        boundaries = c(0.5, 5), # 
                        hysteresis = 0.05, 
                        samplefreq = 100){
  
  # Check whether an AccData obejct 
  # If Not an AccData object is it a numerical vector (Can be timestamps and a vector)
  if (class(AccData) == "AccData"){
    StepData = AccData$data.out[,3]
    samplefreq = AccData$freq
  } else if (class(AccData) == "numeric"){
    StepData = AccData
    if (missing(samplefreq)){
      warning("No samplefreq is given. samplefreq set to default, samplefreq = 100")
      samplefreq = 100
    }
  } else if (class(AccData) == "matrix" & dim(AccData)[2] == 2){
    StepData = AccData[,2]
    if (missing(samplefreq)){
      warning("No samplefreq is given. samplefreq set to default, samplefreq = 100")
      samplefreq = 100
    }
  } else {
    stop("Step Counter must use either an AccData object, Numerical Vector or a 2D Matrix of time and StepData")
  }
  
  Filter <- cheby1(n = filterorder,                               # order of filter
                   Rp = Rp,                                       # ripple of passband
                   W = boundaries/samplefreq,                     # lower then upper frequencies of bandpass
                   type = "pass",
                   plane = "z")
  
  #### Apply the bandpass filter ####
  filteredData = signal::filter(Filter, StepData) 
  
  state = -1                                                       # initialise step state
  interval = 0                                                     # initialise the interval counter
  cadence = numeric(0)                                             # initialise first element of array for intervals
  samples = length(filteredData)                                   # loop through all samples
  
  for (a in 1:samples) {
    if ((filteredData[a] > hysteresis) && (state < 0)){            # new step started
      state = 1                                                    # set the state
      cadence[length(cadence)] = interval +1                       # write the step interval
      cadence[length(cadence)+1] = 0                               # initialise to record the next step    
      interval = 0                                                 # reset the step counter
    } else if ((-1*filteredData[a] > hysteresis) && (state > 0)) { # hysteresis reset condition met
      state = -1                                                   # reset the state
      interval = interval + 1                                      # increment the interval
    } else {
      interval = interval + 1                                      # increment the interval
    }
    cadence[length(cadence)] = interval                            # capture last part step
  }
  
  cadence = cadence/samplefreq                                     # divide by the sample frequency to get seconds
  
  return(cadence)
  
}

# Test script 1
Steps1 = StepCounter(accdata)
length(Steps1)
mean(Steps1)
sd(Steps1)
plot(Steps1)

# Test script 2
vectordata = accdata$data.out[,3]
Steps2 = StepCounter(vectordata)
length(Steps2)
mean(Steps2)
sd(Steps2)
plot(Steps2)

# Test script 3
Steps3 = StepCounter(accdata, 
                     filterorder = 2,
                     Rp = 3,
                     boundaries = c(0.005, 0.05),
                     hysteresis = 0.05, 
                     samplefreq = 100)
length(Steps3)
mean(Steps3)
sd(Steps3)
plot(Steps3)


# Test script 4
d1 <- sin(seq(0.1, 100, 0.1))/2 + rnorm(1000)/10 + 1
Steps4 = StepCounter(d1)
length(Steps4)
mean(Steps4)
sd(Steps4)
plot(Steps4)


funct<-function(filteredData) {
state = -1  # initialise step state
interval = 0  # initialise the interval counter
cadence = numeric(0)  # initialise first element of array for intervals
samples = length(filteredData)  # loop through all samples

for (a in 1:samples) {
if ((filteredData[a] > hysteresis) & & (state < 0)){  # new step started
state = 1  # set the state
cadence[length(cadence)] = interval +1  # write the step interval
cadence[length(cadence)+1] = 0  # initialise to record the next step
interval = 0  # reset the step counter
} else if ((-1 * filteredData[a] > hysteresis) & & (state > 0)) {  # hysteresis reset condition met
state = -1  # reset the state
interval = interval + 1  # increment the interval
} else {
interval = interval + 1  # increment the interval
}
cadence[length(cadence)] = interval  # capture last part step
}

cadence = cadence / 100
return(cadence)
}
"""
from rpy2 import robjects
from rpy2.robjects.packages import importr
import numpy as np
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

GENEAread = importr('GENEAread')
signal = importr('signal')
R = robjects.r
R('memory.limit()')
R('memory.limit(size = 10000)')
R('memory.limit()')

# Read the binfile using R
robjects.reval('q = read.bin("C:/Users/x/Downloads/Raw data walk/jl_right wrist_024600_2017-08-16 17-43-43_walk.bin")')
robjects.reval('p = as.matrix(q$data.out)')
robjects.reval("st = as.character(q$header['Start_Time',][1])")
file_name = robjects.reval('f = as.character(q$filename)')[0]
frequency = int(float(robjects.reval('f = as.character(q$header["Measurement_Frequency",][1])')[0]))

# Create Y-axis as Python array
A = np.asarray(robjects.r.p)

# Build a Cheby filter in R
robjects.reval('Filter = cheby1(n=2, Rp = 3, W = c(0.5, 5)/100, type = "pass", plane = "z")')
robjects.reval('data = p[,3]')
robjects.reval('filteredData = signal::filter(Filter, p[,3])')
RFiltered = np.asarray(robjects.r.filteredData)

# Run the for loop cadence computer in R
string = """
funct<-function(filteredData, hyst, freq) {
state = -1  # initialise step state
interval = 0  # initialise the interval counter
cadence = numeric(0)  # initialise first element of array for intervals
samples = length(filteredData)  # loop through all samples
hysteresis = hyst

for (a in 1:samples) {
if ((filteredData[a] > hysteresis) && (state < 0)){  # new step started
state = 1  # set the state
cadence[length(cadence)] = interval +1  # write the step interval
cadence[length(cadence)+1] = 0  # initialise to record the next step
interval = 0  # reset the step counter
} else if ((-1 * filteredData[a] > hysteresis) && (state > 0)) {  # hysteresis reset condition met
state = -1  # reset the state
interval = interval + 1  # increment the interval
} else {
interval = interval + 1  # increment the interval
}
cadence[length(cadence)] = interval  # capture last part step
}

cadence = cadence / freq
return(cadence)
}
"""

RCode = SignatureTranslatedAnonymousPackage(string, "RCode")
RCadence = np.array(RCode.funct(robjects.r.filteredData, hyst = 0.05, freq = frequency))

#Build cheby filter in Python

from scipy.signal import cheby1, lfilter

def cheby_bandpass(lowcut, highcut, fs, order=2):
        #nyq = 0.5 * fs 
        nyq = fs # This is technically wrong but I need it to match the R code
        low = lowcut / nyq
        high = highcut / nyq
        sos = cheby1(order,rp= 3, Wn =[low, high], btype='bandpass', analog=False)
        return sos

def cheby_bandpass_filter(data, lowcut, highcut, fs, order=2):
        sos = cheby_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(sos[0],sos[1], data)
        return y

PythonFiltered = cheby_bandpass_filter(A, lowcut=0.5, highcut=5, fs = frequency, order=2)


# Build the cadence counter in Python
def compute_Cadence(data, hyst, freq):
	state = -1
	interval = 0
	cadence = []
	for i in range(data.shape[0]):
		if data[i] > hyst and state<0:
			state = 1
			cadence.append((interval+1)/freq)
			interval = 0
		elif (-1*data[i]) > hyst and state>0:
			state = -1
			interval = interval+1
		else:
			interval = interval+1
	cadence.append(interval/freq)
	return cadence[1:]


PythonCadence = np.array(compute_Cadence(PythonFiltered, hyst = 0.05, freq = 100))

# Test for equality in R and Python for both the bin file and arbitrary data
if np.sum(RCadence - PythonCadence)==0:
	print('Values equivalent')

for i in range(312):
	rand = np.random.random(250000)*2 -1
	res = robjects.FloatVector(rand)
	PythonCadence = np.array(compute_Cadence(rand, hyst = 0.5, freq = 100))
	RCadence = np.array(RCode.funct(res, hyst = 0.5, freq = 100))
	if np.sum(PythonCadence-RCadence)!=0:
		print('Error')
		break
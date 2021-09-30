from rpy2 import robjects
from rpy2.robjects.packages import importr
from collections import Counter
from pyunicorn.timeseries.recurrence_plot import RecurrencePlot as rp
import glob
import scipy
import scipy.stats
from datetime import datetime, timedelta


def proc(bin, Act, name):
GENEAread = importr('GENEAread')
R = robjects.r
R('memory.limit()')
R('memory.limit(size = 100000)')
R('memory.limit()')

robjects.reval('q = read.bin("' + bin +'")')
robjects.reval('p = as.matrix(q$data.out)')
frequency = int(float(robjects.reval('f = as.character(q$header["Measurement_Frequency",][1])')[0]))
if frequency !=100:
	print('ERROR')
		#return(3)

A = np.asarray(robjects.r.p)[:, [0, 1, 2, 3]]	
	Labs = np.zeros(A.shape[0])-1
	Steps = np.zeros(A.shape[0])-1
	

	
def barcode(enmo_arr, int_threshold1, int_threshold2, int_threshold3,
 dur_threshold1, dur_threshold2, dur_threshold3, dur_threshold4):
	
	barcode = np.zeros(enmo_arr.shape[0])
	
	int_codes = generic_coder(enmo_arr, int_thresholds)
	consec_int = np.array([(sum( 1 for _ in group ),key) for key, group in itertools.groupby(int_codes)])
		
	dur_codes = np.array(list(map(lambda x:dur_coder(x[0], x[1], dur_thresholds, dur_indexes), consec_int)))
	bouts = np.array(np.vstack((consec_int[:, 0], np.cumsum(consec_int[:, 0]), dur_codes)).T, dtype = np.int)
	# Code durs and int into uniquue values
	
	bouts[:, 0] = bouts[:, 1]-bouts[:,0]
	for b in bouts:
		barcode[b[0]:b[1]] = b[2]
	return barcode




def generic_coder(arr, list_of_thresholds):
	is_greater_than_thresholds = np.apply_along_axis(lambda x: np.greater(x, list_of_thresholds), 1, arr[:, None])
	return np.array([np.max(np.arange(0, len(list_of_thresholds))[x]) for x in is_greater_than_thresholds])
	
def dur_coder(duration,index, dur_thresholds, dur_indexes):
    thresholds = dur_thresholds[index]
    dur_index = np.max(np.arange(len(thresholds))[np.greater(duration, thresholds)])
    return dur_indexes[index]+dur_index
	
int_thresholds = [0, 0.15, 0.25, 0.35]
dur_thresholds = [[0, 30, 60],
[0, 10, 30, 300],
[0, 10, 30, 300],
[0, 10, 30, 300]]

dur_indexes = np.cumsum([len(x) for x in dur_thresholds])
dur_indexes = np.concatenate(([0], dur_indexes[:-1]))


fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
fig.suptitle('Person 590')
ax1.plot(ts[25200+(3600*4):25200+(3600*5)])
ax2.imshow(ts[25200+(3600*4):25200+(3600*5)])

barprops = dict(aspect='auto', cmap='plasma', interpolation='nearest')
fig, ax = plt.subplots()
im = ax.imshow(b.barcode[:, 1].reshape((1, -1)), **barprops)
x_ticks = np.arange(0, b.barcode.shape[0],b.barcode.shape[0]//10)
x_labels = md.num2date(md.epoch2num(b.barcode[:, 0][x_ticks]))
plt.xticks(x_ticks, np.array([x.strftime('%x %X') for x in x_labels]), rotation = '90')
plt.colorbar(im)
plt.show()

# Hide x labels and tick labels for all but bottom plot.
ax1.label_outer()
plt.show()

## Plot for 1 hour
fig, ax = plt.subplots()
x = np.arange(3600)
plot1 = ts[25200+(3600*4):25200+(3600*5)]
extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
im = ax.imshow(plot1[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)

ax.set_xlim(extent[0], extent[1])

ax2.plot(x,plot1)

ax2.set_xticks(np.arange(0,3601, 900))
ax2.set_xticklabels(['07:00','07:15','07:30','07:45','08:00'])
fig.colorbar(im)
ax.set_yticks([])
ax.set_xticks([])
plt.title('Person 590')
ax.set_yticks([])
plt.tight_layout()
plt.show()

#plot 3 of differnt person

fig, (ax2,ax) = plt.subplots(nrows=2)
x = np.arange(3600)
plot3 = ts[25200+(3600*8):25200+(3600*9)]
extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
im = ax.imshow(plot3[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
ax.set_yticks([])
ax.set_xlim(extent[0], extent[1])
ax2.plot(x,plot3)
ax2.set_xticks(np.arange(0,3601, 900))
ax2.set_xticklabels(['15:00','15:15','15:30','15:45','16:00'])
fig.colorbar(im)
ax.set_yticks([])
ax.set_xticks([])
plt.title('Person 405')
ax.set_xlim(extent[0], extent[1])
plt.tight_layout()
plt.show()


## Plot for 6 hours
fig, (ax2,ax) = plt.subplots(nrows=2)
x = np.arange(3600*6)
plot2 = ts[25200+(3600*4):25200+(3600*10)]
extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
im = ax.imshow(plot2[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
ax.set_yticks([])
ax.set_xlim(extent[0], extent[1])

ax2.plot(x,plot2)

ax2.set_xticks(np.arange(0,3601*6, 3600))
ax2.set_xticklabels(['07:00','08:00','09:00','10:00','11:00','12:00','13:00'])
fig.colorbar(im)
ax.set_yticks([])
ax.set_xticks([])
plt.title('Person 590')
plt.tight_layout()
plt.show()


	filtered = cheby_bandpass_filter(Y.flatten(), lowcut=FD, highcut=FU, fs = 30, order=2)
	hy = 0.1
	cadence = compute_Cadence2(filtered, hy, 30)
	zs = np.zeros(Y.flatten().shape[0])
	zs[np.array(np.cumsum(cadence*30), dtype = np.int)]=1
	
	###
	# minute_by_minute
	# trim both to minutr_by_minute (6000)

	if zs.shape[0]%1800!=0:
		zs = zs[:zs.shape[0]//1800*1800]
	
	zs = zs.reshape(zs.shape[0]//1800, 1800)
	
	##MEDIAN
	cad_acc = np.array([(np.median(np.diff(np.where(i)[0])), len(np.where(i)[0])) for i in zs])

	# set all cadences<3 = 0
	cad_acc[cad_acc[:, 1] < 3, 0] = 0

	
	# Set all nans to zeros
	where_are_NaNs = np.isnan(cad_acc)
	cad_acc[where_are_NaNs] = 0
	
	cad_acc_1 = cad_acc
	###### 200
	FU = 5
	FD = 0.5
	fs =30
	filtered = cheby_bandpass_filter(Y.flatten(), lowcut=FD, highcut=FU, fs = 30, order=2)
	hy = 0.22
	cadence = compute_Cadence2(filtered, hy, 30)
	zs = np.zeros(Y.flatten().shape[0])
	zs[np.array(np.cumsum(cadence*30), dtype = np.int)]=1
	
	###
	# minute_by_minute
	# trim both to minutr_by_minute (6000)

	if zs.shape[0]%1800!=0:
		zs = zs[:zs.shape[0]//1800*1800]
	
	zs = zs.reshape(zs.shape[0]//1800, 1800)
	
	##MEDIAN
	cad_acc = np.array([(np.median(np.diff(np.where(i)[0])), len(np.where(i)[0])) for i in zs])

	# set all cadences<3 = 0
	cad_acc[cad_acc[:, 1] < 3, 0] = 0

	
	# Set all nans to zeros
	where_are_NaNs = np.isnan(cad_acc)
	cad_acc[where_are_NaNs] = 0
	
	cad_acc_2 = cad_acc	

	return cad_acc_1, cad_acc_2
    
    import pandas as pd
import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr
from collections import Counter
import glob
import scipy
from scipy.signal import cheby1, filtfilt, lfilter

def cheby_bandpass(lowcut, highcut, fs, order=2):
        #nyq = 0.5 * fs
        nyq = fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = cheby1(order,rp= 3, Wn =[low, high], btype='bandpass', analog=False)
        return sos

def cheby_bandpass_filter(data, lowcut, highcut, fs, order=2):
        sos = cheby_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(sos[0],sos[1], data)
        return y
		
def compute_Cadence2(data, hyst, freq):
	Hyster = (data>hyst)+0.0 - (data <hyst*-1)
	NonZero = np.nonzero(Hyster)[0]
	Cadence = np.hstack((NonZero[0], NonZero[np.where(np.diff(Hyster[np.nonzero(Hyster)[0]])==2)[0]+1], NonZero[-1]))
	return Cadence
					
   

def valid_hours(data, temp, freq):
    day_aggregate = data
    day_temp = temp
    valid = np.zeros((day_aggregate.shape[0] - 120 * freq) // freq)
    t0 = 26
    for i in list(range(0, (day_aggregate.shape[0] - 120 * freq), freq))[:-1]:
       t1 = np.mean(day_temp[i: i + (60 * freq)])
       t2 = np.mean(day_temp[i + (60 * freq):i + (120 * freq)])
       sd_x = np.std(day_aggregate[i + (60 * freq):i + (120 * freq), 0])
       sd_y = np.std(day_aggregate[i + (60 * freq):i + (120 * freq), 1])
       sd_z = np.std(day_aggregate[i + (60 * freq):i + (120 * freq), 2])
       range_x = np.ptp(day_aggregate[i + (60 * freq):i + (120 * freq), 0])
       range_y = np.ptp(day_aggregate[i + (60 * freq):i + (120 * freq), 1])
       range_z = np.ptp(day_aggregate[i + (60 * freq):i + (120 * freq), 2])
       valid[i // freq] = valid_window(t0, t1, t2, sd_x, sd_y, sd_z, range_x, range_y, range_z)
    valid = np.repeat(valid, freq)
    return np.concatenate((valid, np.zeros(data.shape[0] - valid.shape[0])))


def valid_window(t0, t1, t2, sd_x, sd_y, sd_z, range_x, range_y, range_z):
    non_wear_acc = (sd_x < 0.013 and range_x < 50) + (sd_y < 0.013 and range_y < 50) + (sd_z < 0.013 and range_z < 50)
    if t2 < t0 and non_wear_acc > 1.0:
        return 0
    elif t2 >= t0:
        return 1
    else:
        if t2 >= t1:
            return 1
        else:
            return 0

def valid(valid_data):
    return np.mean(valid_data)*24#
    
    
def from_excel_ordinal(ordinal, epoch=datetime(1900, 1, 1)):
	if ordinal > 59:
		ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!
	inDays = int(ordinal)
	frac = ordinal - inDays
	inSecs = int(frac * 86400.0)
	inMicro = round((frac*86400 - int(frac*86400)) *1000000)
	return epoch + timedelta(days=inDays - 1, seconds=inSecs, microseconds = inMicro) # epoch is day 1
	
    
 def events(data):
    i=0
    store = []
    while i<len(data):
        j=0
        if data[i] >=0.4:
            arr = data[i]
            while np.mean(arr)>=0.32:
                j = j + 1
                
                if (i+j)>len(data):
                    store.append([i, j-1])
                    return store
                    
                arr = data[i:(i+j)]
            store.append([i, j-1])
            i = i + j
        else:
            i = i + 1
    return i, store
    
    
concat_all_files
np.conc
    
from sklearn.metrics import classification_report as cr

u = np.array([x.pal_steps.shape[0] - x.step_arr_acc.shape[0] for x in v_days])
u = u>-1
v_days = v_days[u]
total_minutes_acc  =  np.concatenate([x.step_arr_acc for x in v_days])
total_minutes_pal  =  np.concatenate([x.pal_steps for x in v_days])
    
    
#With valid day computation, then renmove the non walking from the acc daya
total_minutes_acc[total_minutes_acc<20] = 0
"""
20-50-

Not walking<20
Slow walking>20-50
Fast walking>50
"""
def labs2(val, x):

	if val<x[0]:
		return 0
	if val<x[1]:
		return 1
	return 2
    
classes_acc = np.array(list(map(lambda x: labs2(x, [22.5, 50]), total_minutes_acc)))
classes_pal = np.array(list(map(lambda x: labs2(x, [22.5, 50]), total_minutes_pal)))



import numpy as np
for i in np.arange(30):   
   size = 4000
    arr = np.random.random(size)
    size = size+4000 - np.sum(arr<0.1)




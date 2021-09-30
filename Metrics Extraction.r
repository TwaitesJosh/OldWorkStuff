install.packages('lubridate')
library(lubridate)

install.packages('TSEntropies')
library(TSEntropies)

install.packages("entropy")
library(entropy)

# Add bouts that have blank values at XX:00:01
ghost_bouts<-function(row){

# Blank all values in row except Date, TS, Day and Nonwear
row[head(colnames(row), 26)]=0

# Manaully set the first blank tab to '00:00:01'
row['Time stamp'] = '00:00:01'
blank_tab = row

# Iterate over 23 hours
for (i in seq(2,23)){
row['Time stamp'] = paste(sprintf("%02d",i), ':00:01', sep = '')
blank_tab = rbind(blank_tab, row)
}
return(blank_tab)
}

# Identify the periods of activty greater that the threshold
activityduration<-function(input,threshold, allowance){

input = c(input,-1)

j=1
i = 0
counter = 0
tuples1 = numeric(length(input)/2)
tuples2 = numeric(length(input)/2)
while (input[j] !=-1) {

if (input[j] >= threshold){

	k=j
	
	while (mean(input[j:(k+i)])>=threshold*allowance){
		i= i+1
		
		if (input[k+i] == -1){
			tuples1[counter+1] = j
			tuples2[counter+1] = (k+i-1)
			
			return(data.frame(tuples1,tuples2))}
	}
	
	counter = counter+1
	tuples1[counter] = j
	tuples2[counter] = (k+i)-1
	j =k+i
	i=0
	
	}
	j = j+1
}

return(data.frame(tuples1,tuples2))
}

# Compute quantiles for events
event_quantile<-function(arr){
	return(quantile(arr, seq(0.05, 0.95,0.05)))
	}

# Compute volume for events
event_vol<-function(arr){
	return(sum(arr))
	}
	
# Compute trimmed SD for events	
trimmed_sd<-function(arr){
	bottom = quantile(arr, 0.05)
	top = quantile(arr, 0.95)
	return(sd(arr[(arr<top & arr>bottom)]))
}
# Compute date for events	
event_date<-function(arr){
	return(as.Date(arr))
	}
# Compute day of week for events	
event_day_of_week<-function(arr){
	return(wday(as.Date(arr), label=True))
	}

# Compute day of wear	
event_day_of_wear<-function(arr){
	day_nums = yday(as.Date(arr))
	return(match(day_nums, unique(day_nums)))
	}
	
# function for turning ENMO_data and TS data into events
extract_table<-function(ENMO_data, ts_data,dur_threshold ,threshold, percent)
{

threshold_data<- pmin(ENMO_data, threshold)
# Extract event times
times<-activityduration(as.numeric(threshold_data),threshold,percent)

# Remove events less than duration threshold
times = times[times[,1]!=0,]
times = times[(times[,2]-times[,1])>=dur_threshold,]

# Rename columns
names(times) <- c('Start', 'End')

# Create list of ENMO arrays for each event
events = apply(times, 1, function(x) ENMO_data[x[1]:x[2]])

#Volume
times['Volume'] = sapply(events, sum)

# Duration
times['Duration'] = sapply(events, length)

#Intesity
times['Intensity'] = times['Volume']/times['Duration']

#Percentiles
percentiles = lapply(events, quantile, probs = seq(0.05,0.95,0.05))
percentile_columns = paste(seq(5,95,5), 'th percentile', sep='')
times[percentile_columns] =  t(data.frame(percentiles))

# SD
times['Standard deviation'] = sapply(events,sd)
times['Trimmed SD'] = sapply(events,trimmed_sd)

# Date/Time
times['Date'] = as.Date(ts_data[times[,'Start']])
times['Time stamp'] = substr(ymd_hms(ts_data[times[,'Start']]),12,19)

#Day of week
times['Day of Week'] = wday(as.Date(ts_data[times[,'Start']]), label=TRUE)

# Day number
day_nos = yday(as.Date(ts_data[times[,'Start']]))
times['Day of Wear'] = match(day_nos, unique(day_nos))

# Nonwear
valid_hours = aggregate(pmin(M$metalong['nonwearscore'], 1.9)+(1/96), by = list(as.Date(M$metalong[,1])), sum, na.rm=TRUE)
times['Nonwear'] = merge(times, valid_hours, by.x = 'Date', by.y = 'Group.1')['nonwearscore']


# Median volumne
half_vol = sum(sort(times[,'Volume']))/2
median_vol = sort(times[,'Volume'])[cumsum(sort(times[,'Volume']))>half_vol][1]
times['Med_vol'] = median_vol


# Median duration
half_dur = sum(sort(times[,'Duration']))/2
median_dur = sort(times[,'Duration'])[cumsum(sort(times[,'Duration']))>half_dur][1]
times['Med_dur'] = median_dur


# Median Intensity
hal_int = sum(sort(times[,'Intensity']))/2
median_int = sort(times[,'Intensity'])[cumsum(sort(times[,'Intensity']))>hal_int][1]
times['Med_int'] = median_int

# Barcodes
daily_barcodes = aggregate(ENMO_data, by = list(as.Date(ts_data)), barcode)
daily_barcodes = do.call(data.frame, daily_barcodes)
barcode_columns = c('Entropy', 'Sample Entropy', 'LZC Entropy', 'Percent Active')
colnames(daily_barcodes)<- c('Group.1', barcode_columns)
times[barcode_columns] = merge(times, daily_barcodes, by.x = 'Date', by.y = 'Group.1')[barcode_columns]


# Identify the days of the week that need 'ghost bouts'
to_ghost = rownames(unique(times['Day of Wear']))

for (i in to_ghost){
times = rbind(times, ghost_bouts(times[i,]))
}
# Add ID
times['ID'] = I$filename
return(times)
}

# Create PA barcodes from ENMO data
barcode<-function(ENMO_in){

int_cutoff = c(0.04, 0.08, 0.10)
cutoff = c(-Inf,5,10,60, Inf)

cuts <- c(-Inf, 0.04, 0.08, 0.10, Inf)
labs <- c("Sed", "Low", "Med", "High")

barcode_lab = labs[findInterval(ENMO_in, cuts)]
barcode_rle = rle(barcode_lab)

len = barcode_rle$'lengths'
act = barcode_rle$'values'

new_act_labs = rep(0, length(act))

# Reallocate barcode labels
for (i in seq_along(labs)){
	lab_location = barcode_rle$'values'==labs[i]
	new_labs = seq(((i-1)*4)+1,(4*i))[findInterval(barcode_rle$'length'[lab_location], cutoff)]
	new_act_labs[lab_location] = new_labs
	}

barcode_rle$'values' = new_act_labs
new_barcode = inverse.rle(barcode_rle)


entropy = entropy(sapply(unique(new_barcode), function(x) sum(new_barcode==x)))
sampleEntropy = FastSampEn(new_barcode, dim=3, lag = 1, r=1)
LZC_entropy = LZC(as.character(new_barcode))
percent_active = mean(new_barcode>4)

return(c(entropy, sampleEntropy, LZC_entropy, percent_active))
}

# compute LZC complexity from barcode
LZC<-function(input_string){
dict_size = 20
dict = as.character(seq(1,20))

w = ""
result = c()

for (char in input_string){
	wc = paste(w, char, sep='')
	
	if (wc %in% dict){
		w = wc
	}
	else{
	result = c(result, w)
	dict = c(dict, wc)
	dict_size = dict_size+1
	w=char
	}
}
result = c(result, w)
top = length(unique(result))
bottom = length(input_string)/log(length(input_string), length(unique(input_string)))
return(top/bottom)
}

# Actual main function to be run, takes a file ID and all thresholds

load_and_extract<-function(f, dur_threshold, threshold, percent){
load(f,.GlobalEnv)
ENMO_data <-as.numeric(as.character(M$metashort[,2]))
ts_data = as.character(M$metashort[,1])	
tab = extract_table(ENMO_data, ts_data,dur_threshold ,threshold, percent)
filename = paste(f, '.csv', sep='')
write.csv(tab, filename)
}

# Runs the extraction over the file lists
# Catches any erros that occour
extract_all<-function(file_loc, file_list, dur_threshold, threshold, percent){
file_list = paste(file_loc, files, sep = '\\')
for (f in file_list){
  skip_to_next <- FALSE
  # Note that print(b) fails since b doesn't exist

  tryCatch(load_and_extract(f, dur_threshold, threshold, percent), error = function(e) { skip_to_next <<- TRUE})

  if(skip_to_next) { next }     

}
}

# example run
files <- list.files('C:\\Users\\x\\Downloads\\OneDrive_2021-07-01\\Jones T2D data')
extract_all('C:\\Users\\x\\Downloads\\OneDrive_2021-07-01\\Jones T2D data', files, 10, 0.04, 0.80)




# Combine all files into one massive CSV for further analysis
combine_all<-function(file_loc){
csv_files <- list.files(file_loc, pattern = '*.csv')
csv_list = paste(file_loc, csv_files, sep = '\\')
for (data in csv_list){
  
  # Create the first data if no data exist yet
  if (!exists('All_files')){
    All_files <- read.csv(data, header=TRUE)
  }
  
  # if data already exist, then append it together
  if (exists('All_files')){
    tempory <-read.csv(data, header=TRUE)
    All_files <-rbind(All_files, tempory)
    rm(tempory)
  }
}
All_files['ID_number'] = match(as.character(All_files[,'ID']),unique(as.character(All_files[,'ID'])))
return(All_files)
}


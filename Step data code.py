import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import r2_score as r2
import warnings
import numpy
import itertools

def rle(bits):
  # make sure all runs of ones are well-bounded
  bounded = numpy.hstack(([0], bits, [0]))
  # get 1 at run starts and -1 at run ends
  difs = numpy.diff(bounded)
  run_starts, = numpy.where(difs > 0)
  run_ends, = numpy.where(difs < 0)
  return run_starts, run_ends - run_starts




# import file
tab = pd.read_csv('C:/Users/x/Documents/New folder/Merged_files_steps.csv', low_memory=False)
s = np.load('C:/Users/x/Documents/New folder/All_step data.npy', allow_pickle = True)
## Dependent varaibles-direct:
"""


Final Score
MAT score
PASE outside walking time

## Dependent variables to be computed
Z-scores of 4M walk
Z_scores of CR 
Z_scores grip
Average of 3 above z-scores
"""
# Generate FMWS score
tab['M00_SPPB_FMWSTime'] = tab[['M00_SPPB_FMWSAttempt2Time', 'M00_SPPB_FMWSAttempt1Time']].min(1)
# Outlier in FMWS set to nan


def pred_encode(arr1, arr2, tuple):
	
	encode_arr1 = encoder(arr1)
	encode_arr2 = encoder(arr2)
	
	if tuple[4]=='0':
		for i in np.arange(4):
			if tuple[i]=='1':
				encode_arr1[encode_arr2==i]=i
		return encode_arr1
	if tuple[4]=='1':
		for i in np.arange(4):
			if tuple[i]=='1':
				encode_arr2[encode_arr1==i]=i
		return encode_arr2
	
	
def encoder(arr1):
	ret = np.zeros(arr1.shape[0])
	for i in np.arange(arr1.shape[0]):
		if arr1[i]<10:
			ret[i] = 0
		elif arr1[i]<20:
			ret[i]=1
		elif arr1[i]<40:
			ret[i] = 2
		else:
			ret[i] = 3
	return ret
	
	
def shredder(arr_step, arr_class):
	time_in_0 = np.mean(arr_class==0)
	time_in_1 = np.mean(arr_class==1)
	time_in_2 = np.mean(arr_class==2)
	time_in_3 = np.mean(arr_class==3)
	step_in_1 = np.sum(arr_step[arr_class==1])
	step_in_2 = np.sum(arr_step[arr_class==2])
	step_in_3 = np.sum(arr_step[arr_class==3])
	return np.array([time_in_0, time_in_1, time_in_2, time_in_3, step_in_1, step_in_2, step_in_3])
	
def ids(full_id):
	return full_id.split('/')[-1].split('_')[0]
	
ui = [np.concatenate(([ids(x[2])], shredder(x[0][:, 1],encoder(x[0][:, 1])))) for x in s]

step_tab = pd.DataFrame(ui, columns = ['Small_ID', 'time_in_0', 'time_in_1', 'time_in_2', 'time_in_3', 'steps_in_1', 'steps_in_2','steps_in_3'])

sup_tab = pd.merge(tab, step_tab, left_on = 'ID', right_on = 'Small_ID')


def mean_bout_dur(classes, steps, clas, dur):
	#classes = encoder(steps)
	bouts_loc, bouts_len = rle(keep_class(classes,clas))
	
	bouts_loc_removed = bouts_loc[bouts_len>dur]
	bouts_len_removed = bouts_len[bouts_len>dur]	
	mean_dur = np.mean(bouts_len_removed)
	mean_steps = np.mean([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc_removed, bouts_len_removed)])
	return (mean_dur, mean_steps)

def bout_num(classes, steps, clas, dur):
	#classes = encoder(steps)
	bouts_loc, bouts_len = rle(keep_class(classes,clas))
	
	bouts_loc_removed = bouts_loc[bouts_len>dur]
	bouts_len_removed = bouts_len[bouts_len>dur]	
	
	return bouts_len_removed.shape[0]
	
def percent_in_bouts(classes, steps, clas, dur):
	#classes = encoder(steps)
	bouts_loc, bouts_len = rle(keep_class(classes,clas))
	
	bouts_loc_removed = bouts_loc[bouts_len>dur]
	bouts_len_removed = bouts_len[bouts_len>dur]	
	
	dur_perc_of_bouts = bouts_len_removed.shape[0]/(bouts_len.shape[0] + 1)
	dur_perc_of_total = bouts_len_removed.shape[0]/steps.shape[0]
	
	total_kept_steps = np.sum([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc_removed, bouts_len_removed)])
	total_all_steps = np.sum([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc, bouts_len)])
	
	steps_perc_of_bout = total_kept_steps/(total_all_steps+0.0)
	steps_perc_of_total = total_kept_steps/(np.sum(steps)+0.0)
	
	return steps_perc_of_bout, steps_perc_of_total, dur_perc_of_bouts, dur_perc_of_total


def fragmentation(classes, steps, clas, dur):
	#classes = encoder(steps)
	bouts_loc, bouts_len = rle(keep_class(classes,clas))
	
	bouts_loc_removed = bouts_loc[bouts_len>dur]
	bouts_len_removed = bouts_len[bouts_len>dur]	
	
	N = bouts_len_removed.shape[0]
	if N==0:
		return 0, 0, 0, 0
	x_min = np.min(bouts_len_removed)
	
	alpha = (N*(np.sum(np.log(bouts_len_removed/x_min))**-1))+1
	x_1_2 = 2**(1.0/((alpha-1)))*x_min
	W_1_2 = np.sum(bouts_len_removed[bouts_len_removed>x_1_2])/np.sum(bouts_len_removed)
	
	mad = np.abs(np.subtract.outer(bouts_len_removed, bouts_len_removed)).mean()
	g = 0.5* (mad/bouts_len_removed.mean())
	
	return alpha, x_1_2, W_1_2, g

def keep_class(arr, to_keep):
	if to_keep ==0:
		return arr==0
	if to_keep ==1:
		return arr==1
	if to_keep ==2:
		return arr==2
	if to_keep ==3:
		return arr==3
	if to_keep ==(0,1):
		return np.logical_or(arr==0, arr==1)
	if to_keep ==(0,2):
		return np.logical_or(arr==0, arr==2)
	if to_keep ==(0,3):
		return np.logical_or(arr==0, arr==3)
	if to_keep ==(1,2):
		return np.logical_or(arr==2, arr==1)
	if to_keep ==(1,3):
		return np.logical_or(arr==3, arr==1)
	if to_keep ==(2,3):
		return np.logical_or(arr==2, arr==3)
	if to_keep ==(0,1,2):
		return arr!=3
	if to_keep ==(0,1,3):
		return arr!=2
	if to_keep ==(0,2,3):
		return arr!=1
	if to_keep ==(1,2,3):
		return arr!=0	
		

classes_to_keep = [1,2,3,(0,1), (0,2), (0,3), (1,2), (1,3), (2, 3), (0,1,2), (0,1,3), (0,2,3), (1,2,3)]


encoded = np.array([encoder(x[0][:, 1]) for x in s])
print('merged')
frag_tables = np.zeros((755, 224))

for i in range(56):

	#bouts_loc, bouts_len = rle(keep_class(classes,clas))
	frag_tables[:, (i*4):((i+1)*4)] = np.array([fragmentation(y,x[0][:, 1], l[i][0], l[i][1]) for x, y in zip(s, encoded)])
	print(i)
	
perc_tables = np.zeros((755, 224))
for i in range(56):
	perc_tables[:, (i*4):((i+1)*4)] = np.array([percent_in_bouts(y,x[0][:, 1], l[i][0], l[i][1]) for x, y in zip(s, encoded)])
	print(i)

bout_num_tables = np.zeros((755, 56))
for i in range(56):
	bout_num_tables[:, i] = np.array([bout_num(y,x[0][:, 1], l[i][0], l[i][1]) for x, y in zip(s, encoded)])
	print(i)

mean_tables = np.zeros((755, 112))
for i in range(56):
	mean_tables[:, (i*2):((i+1)*2)] = np.array([mean_bout_dur(y,x[0][:, 1], l[i][0], l[i][1]) for x, y in zip(s, encoded)])
	print(i)

mean_tables = np.zeros((755, 117))
for i in range(13):
	mean_tables[:, (i*9):((i+1)*9)] = np.array([np.concatenate((all_funcs(y,x[0][:, 1], classes_to_keep[i], 0), [ids(x[2])])) for x, y in zip(s, encoded)])
	print(i)
	

names = ['mean_dur', 'mean_dur_removed', 'mean_steps', 'mean_steps_removed', 'std_dur', 'std_dur_removed', 'std_steps', 'std_steps_removed', 'ID']
col_names = []
for i in classes_to_keep:
	col_names.append([x + '_' + str(i) for x in names])

col_names = np.array(col_names).flatten()
keep_all = np.ones(len(col_names))
to_remove = np.arange(17, 117, 9)

keep_all[to_remove]=0

means_tables = pd.DataFrame(mean_tables[:,keep_all==1], columns = col_names[keep_all==1])
sup_tab1 = pd.merge(sup_tab, means_tables, left_on = 'Small_ID', right_on = 'ID_1')

cont_variables = [
'M00_BestGrip',
'M00_SPPB_FMWSTime',
'M00_SPPB_CRTime',
'M00_MATSF_MATsfScore']

cont_table = sup_tab1[cont_variables]
# convert all non numercics to nan's
cont_table = cont_table.apply(pd.to_numeric, errors='coerce')
cont_table['M00_SPPB_FMWSTime'][cont_table['M00_SPPB_FMWSTime'].idxmax()] = np.nan
cont_table.describe()
cont_table.skew()
cont_table.kurt()
"""
for i in cont_table.columns:
	u = cont_table[i].values
	u=u[np.logical_not(np.isnan(u))]
	sns.distplot(u, fit=norm);
	plt.title(i)
	plt.show()
	res = stats.probplot(u, plot=plt)
	plt.show()
"""
# Log all tables

log_table = cont_table.apply(np.log)
log_table['M00_MATSF_MATsfScore'] = cont_table['M00_MATSF_MATsfScore']
stan_table = (log_table-log_table.mean())/log_table.std(ddof=0)
stan_table['average Z_score'] = (stan_table['M00_BestGrip']*-1 + stan_table[['M00_SPPB_FMWSTime','M00_SPPB_CRTime']].sum(1)).div(3)
# Insane outlier, obviousy wrong data

interval_vari = np.concatenate((dep_vari[:8],dep_vari[9:25]))

sup_tab_num = sup_tab1.apply(pd.to_numeric, errors='coerce')
for i in dep_vari[:20]:
	u = sup_tab_num[i].values
	u=u[(np.isnan(u) + np.isinf(u))<1]
	low_range = np.mean(u) - np.std(u)*5
	upper_range = np.mean(u) + np.std(u)*5
	if np.sum(u==1)>500:
		continue	
	keep = np.logical_and(u>low_range, u<upper_range)
	sns.distplot(u[keep], fit=norm);
	plt.title(i)
	#plt.show()
	#res = stats.probplot(u[keep], plot=plt)
	plt.show()

to_log = np.append(dep_vari[17:25], ['steps_in_3'])
to_not_log = np.concatenate((dep_vari[:8],dep_vari[9:17]))
logged = (1+sup_tab_num[to_log]).apply(np.log)

sup_tab_num = (sup_tab1.apply(pd.to_numeric, errors='coerce')+1)
X = sup_tab_num[to_not_log]
X[to_log] = logged



X = (X-X.mean())/X.std(ddof = 0)

clean_X = X[X.abs()<3.5]
clean_stan_table = stan_table[(stan_table.abs()<3.5)]
for j in np.arange(25):
	for i in clean_stan_table.columns:
		y = clean_stan_table[i]
		X_in = clean_X[['steps_in_3', clean_X.columns[j]]]

		y = y.apply(pd.to_numeric, errors='coerce')
		to_keep = np.sum(np.isnan(X_in).values, 1)+np.isnan(y).values==0
		X_in = X_in[to_keep]
		y = y[to_keep]
		
		X_in = sm.add_constant(X_in)
		
		t1 = sm.OLS(y, X_in).fit()
		
		
		score1 = t1.rsquared
		
		t2 = sm.OLS(y, X_in[['const', 'steps_in_3']]).fit()
		score2 = t2.rsquared
		if score1 - score2 > 0.05:
			print(t1.summary())
	



def all_funcs(classes, steps, clas, dur):
	
	dur = 1
	bouts_loc, bouts_len = rle(keep_class(classes,clas))
	
	bouts_loc_removed = bouts_loc[bouts_len>dur]
	bouts_len_removed = bouts_len[bouts_len>dur]
	
	mean_dur = np.mean(bouts_len)
	mean_dur_removed = np.mean(bouts_len_removed)
	
	std_steps = np.std([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc, bouts_len)])
	std_steps_removed = np.std([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc_removed, bouts_len_removed)])

	std_dur = np.std(bouts_len)
	std_dur_removed = np.std(bouts_len_removed)
	
	mean_steps = np.mean([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc, bouts_len)])
	mean_steps_removed = np.mean([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc_removed, bouts_len_removed)])
		
	return mean_dur, mean_dur_removed, mean_steps, mean_steps_removed, std_dur, std_dur_removed, std_steps, std_steps_removed
	
	
	
	
	
	dur_perc_of_bouts = bouts_len_removed.shape[0]/(bouts_len.shape[0] + 1)
	dur_perc_of_total = bouts_len_removed.shape[0]/steps.shape[0]
	
	total_kept_steps = np.sum([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc_removed, bouts_len_removed)])
	total_all_steps = np.sum([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc, bouts_len)])
	
	steps_perc_of_bout = total_kept_steps/(total_all_steps+0.0)
	steps_perc_of_total = total_kept_steps/(np.sum(steps)+0.0)


	N = bouts_len_removed.shape[0]
	if N==0:
		return np.zeros(11)
	x_min = np.min(bouts_len_removed)
	
	alpha = (N*(np.sum(np.log(bouts_len_removed/x_min))**-1))+1
	x_1_2 = 2**(1.0/((alpha-1)))*x_min
	W_1_2 = np.sum(bouts_len_removed[bouts_len_removed>x_1_2])/np.sum(bouts_len_removed)
	
	mad = np.abs(np.subtract.outer(bouts_len_removed, bouts_len_removed)).mean()
	g = 0.5* (mad/bouts_len_removed.mean())
	
	return (alpha, x_1_2, W_1_2, g, steps_perc_of_bout, steps_perc_of_total, dur_perc_of_bouts, dur_perc_of_total,bouts_len_removed.shape[0], mean_dur, mean_steps)
	
	
y = sup_tab_num['M00_SPPB_FMWSTime']	
keep = np.isnan(sup_tab_num).sum()<70
sup_tab_num[sup_tab_num.columns[keep]]
sup_tab_num = sup_tab_num.apply(pd.to_numeric, errors='coerce')+1
X = sup_tab_num
X = X.apply(pd.to_numeric, errors='coerce')

X = (X-X.mean())/X.std(ddof=0)

keep = np.isnan(sup_tab_num).sum()<70
X = X[X.columns[keep]]

y = (y-y.mean())/y.std(ddof=0)

for i in X.columns:
	X_in = X[['steps_in_3', i]]
	X_in.abs()>3.5
	y = sup_tab_num['M00_SPPB_FMWSTime']
	y = y.apply(pd.to_numeric, errors='coerce')
	X_in = X_in.apply(pd.to_numeric, errors='coerce')
	to_keep = np.logical_and(np.isnan(y).values==0, (np.isnan(X_in).sum(1)<1).values)

	to_keep[(X_in.abs()>3.5).sum(1).values >0] = False
	
	if np.sum(to_keep) < 300 or np.sum(np.diff(X_in[i]))==0:
		continue
	X_in = X_in[to_keep]
	y = y[to_keep]	
	
	X_in = sm.add_constant(X_in)
	
	
	t = sm.OLS(y, X_in).fit()
	score_1 = t.rsquared
	t1 = sm.OLS(y, X_in.values[:, :-1]).fit()
	score_2 = t1.rsquared
	
	if score_1-score_2 > 0.1 and score_2>0.1:
		print(i)
		
		
def count_med(classes, steps, clas):
	
	bouts_loc, bouts_len = rle(keep_class(classes,clas))
	steps_count = np.array([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc, bouts_len)])
	
	dur = np.median(bouts_len)

	bouts_loc_removed_long = bouts_loc[bouts_len>dur]
	steps_count_removed_long = steps_count[bouts_len>dur]
	bouts_len_removed_long = bouts_len[bouts_len>dur]
	
	bouts_loc_removed_shrt = bouts_loc[bouts_len<=dur]
	steps_count_removed_shrt = steps_count[bouts_len<=dur]
	bouts_len_removed_shrt = bouts_len[bouts_len<=dur]
	

	return bouts_len_removed_long.shape[0], np.sum(steps_count_removed_long),bouts_len_removed_shrt.shape[0], np.sum(steps_count_removed_shrt)
	
med_2 = np.array([count_med(y,x[0][:, 1],2) for x, y in zip(s, encoded)])
med_3 = np.array([count_med(y,x[0][:, 1],3) for x, y in zip(s, encoded)])
med_23 = np.array([count_med(y,x[0][:, 1],(2,3)) for x, y in zip(s, encoded)])

def count_dur(classes, steps, clas):
	
	bouts_loc, bouts_len = rle(keep_class(classes,clas))
	steps_count = np.array([np.sum(steps[x:x+y]) for x,y in zip(bouts_loc, bouts_len)])
	
	bouts_loc_removed_ls5 = bouts_loc[bouts_len<5]
	steps_count_removed_ls5 = steps_count[bouts_len<5]
	bouts_len_removed_ls5 = bouts_len[bouts_len<5]
	
	bouts_loc_removed_ls10 = bouts_loc[np.logical_and(bouts_len>5, bouts_len<10)]
	steps_count_removed_ls10 = steps_count[np.logical_and(bouts_len>5, bouts_len<10)]
	bouts_len_removed_ls10 = bouts_len[np.logical_and(bouts_len>5, bouts_len<10)]
	
	bouts_loc_removed_gr10 = bouts_loc[bouts_len>10]
	steps_count_removed_gr10 = steps_count[bouts_len>10]
	bouts_len_removed_gr10 = bouts_len[bouts_len>10]
	
	to_return = [bouts_loc_removed_ls5.shape[0], np.sum(steps_count_removed_ls5),
	bouts_loc_removed_ls10.shape[0], np.sum(steps_count_removed_ls10),
	bouts_loc_removed_gr10.shape[0], np.sum(steps_count_removed_gr10)]
	
	return np.array(to_return)
	
	
count_2 = np.array([count_dur(y,x[0][:, 1],2) for x, y in zip(s, encoded)])
count_3 = np.array([count_dur(y,x[0][:, 1],3) for x, y in zip(s, encoded)])
count_23 = np.array([np.concatenate((count_dur(y,x[0][:, 1],(2,3)), [ids(x[2])])) for x, y in zip(s, encoded)])	
	
tab = np.hstack((med_2, med_3, med_23, count_2, count_3, count_23))
columns_med_2 = ['med_long_num_2', 'med_long_steps_2', 'med_short_num_2', 'med_short_steps_2']
columns_med_3 = ['med_long_num_3', 'med_long_steps_3', 'med_short_num_3', 'med_short_steps_3']
columns_med_23 = ['med_long_num_23', 'med_long_steps_23', 'med_short_num_23', 'med_short_steps_23']

columns_bouts_2 = ['num_ls_5_2', 'steps_ls_5_2', 'num_ls_10_2', 'steps_ls_10_2','num_gr_10_2', 'steps_gr_10_2']
columns_bouts_3 = ['num_ls_5_3', 'steps_ls_5_3', 'num_ls_10_3', 'steps_ls_10_3','num_gr_10_3', 'steps_gr_10_3']
columns_bouts_23 = ['num_ls_5_23', 'steps_ls_5_23', 'num_ls_10_23', 'steps_ls_10_23','num_gr_10_23', 'steps_gr_10_23', 'ID']

all_cols = np.concatenate((columns_med_2, columns_med_3, columns_med_23, columns_bouts_2, columns_bouts_3, columns_bouts_23))
bout_tab = pd.DataFrame(tab, columns = all_cols)




cont_variables = [
'M00_SPPB_FMWSTime']

cont_table = sup_tab[cont_variables]
# convert all non numercics to nan's
cont_table = cont_table.apply(pd.to_numeric, errors='coerce')
cont_table['M00_SPPB_FMWSTime'][cont_table['M00_SPPB_FMWSTime'].idxmax()] = np.nan
log_table = cont_table.apply(np.log)

stan_table = (log_table-log_table.mean())/log_table.std(ddof=0)

dep_vari = sup_tab.columns[1071:-1]
sup_tab_num = sup_tab.apply(pd.to_numeric, errors='coerce')


to_log = np.append(dep_vari[17:25], ['steps_in_3'])
to_not_log = np.concatenate((dep_vari[:8],dep_vari[9:17]))
logged = (1+sup_tab_num[to_log]).apply(np.log)

sup_tab_num = (sup_tab1.apply(pd.to_numeric, errors='coerce')+1)
X = sup_tab_num[to_not_log]
X[to_log] = logged
X = (X-X.mean())/X.std(ddof = 0)

clean_X = X[X.abs()<3.5]
clean_stan_table = stan_table[(stan_table.abs()<3.5)]
for j in np.arange(25):
	for i in clean_stan_table.columns:
		y = clean_stan_table[i]
		X_in = clean_X[['steps_in_3', clean_X.columns[j]]]

		y = y.apply(pd.to_numeric, errors='coerce')
		to_keep = np.sum(np.isnan(X_in).values, 1)+np.isnan(y).values==0
		X_in = X_in[to_keep]
		y = y[to_keep]
		
		X_in = sm.add_constant(X_in)
		
		t1 = sm.OLS(y, X_in).fit()
		
		
		score1 = t1.rsquared
		
		t2 = sm.OLS(y, X_in[['const', 'steps_in_3']]).fit()
		score2 = t2.rsquared
		if score1 - score2 > 0.05:
			print(t1.summary())
	
	
to_log = ['steps_in_3','M00_SPPB_FMWSTime']

logged = (1+sup_tab_num[to_log]).apply(np.log)
sup_tab_num = (sup_tab.apply(pd.to_numeric, errors='coerce')+1)
X = sup_tab_num[np.concatenate((all_cols[:-1], ['steps_in_3'], ['M00_BestGrip',
'M00_SPPB_CRTime','M00_MATSF_MATsfScore']))]
X[to_log] = logged
X = (X-X.mean())/X.std(ddof = 0)

clean_X = X[X.abs()<3.5]

clean_stan_table = stan_table[(stan_table.abs()<3.5)]

to_pred = ['M00_BestGrip', 'M00_SPPB_FMWSTime', 'M00_SPPB_CRTime','M00_MATSF_MATsfScore']

scores = np.zeros((4, 30, 10), dtype = np.object)

for j in np.arange(30):
	for k, i in enumerate(to_pred):
		y = clean_X[[i]]
		X_in = clean_X[[clean_X.columns[j], 'steps_in_3']]

		y = y.apply(pd.to_numeric, errors='coerce')
		X_keep = np.sum(np.isnan(X_in),1).values!=0
		y_keep = np.isnan(y).values.flatten()
		
		to_keep = np.logical_not(np.logical_or(y_keep, X_keep))
		X_in = X_in[to_keep==1]
		y = y[to_keep==1]
		
		X_in = sm.add_constant(X_in)
		
		t1 = sm.OLS(y, X_in).fit()
		
		
		score1 = t1.rsquared
		
		t2 = sm.OLS(y, X_in[['const', clean_X.columns[j]]]).fit()
		score2 = t2.rsquared
		
		scores[k, j] = np.array([clean_X.columns[j], i, score1, score2, t1.pvalues.values[1], t1.pvalues.values[2],t1.params.values[1], t1.params.values[2], t2.pvalues.values[1],t2.params.values[1]])
			

tab_0 = pd.DataFrame(scores[0], columns = ['indep', 'dep', 'score with both', 'score with dep', 'p value dep both', 'p value steps both', 'coeff dep both', 'coeff step both', 'p value dep only', 'coeff dep only'])		
tab_1 = pd.DataFrame(scores[1], columns = ['indep', 'dep', 'score with both', 'score with dep', 'p value dep both', 'p value steps both', 'coeff dep both', 'coeff step both', 'p value dep only', 'coeff dep only'])		
tab_2 = pd.DataFrame(scores[2], columns = ['indep', 'dep', 'score with both', 'score with dep', 'p value dep both', 'p value steps both', 'coeff dep both', 'coeff step both', 'p value dep only', 'coeff dep only'])		
tab_3 = pd.DataFrame(scores[3], columns = ['indep', 'dep', 'score with both', 'score with dep', 'p value dep both', 'p value steps both', 'coeff dep both', 'coeff step both', 'p value dep only', 'coeff dep only'])		
		
from sklearn.preprocessing import Normalizer as N
from sklearn.decomposition import PCA

X = sup_tab_num[sup_tab_num.columns[(np.isnan(sup_tab_num).sum(0) <20).values]].apply(np.log)
y = sup_tab_num[['M00_MATSF_MATsfScore']].apply(np.log)

drop_cols = np.array([x.rfind('M00') for x in X.columns])
X = X[X.columns[drop_cols!=0]]
X_keep = np.sum(np.isnan(X),1).values!=0
y_keep = np.isnan(y).values.flatten()

to_keep = np.logical_not(np.logical_or(y_keep, X_keep))
X = X[to_keep==1]
y = y[to_keep==1]


norm = SS()
pca = PCA(n_components=4)

norm_X = norm.fit_transform(X)
pca_X = pca.fit_transform(norm_X, y)
X_in = sm.add_constant(pca_X)

t1 = sm.OLS(y, X_in[:, :5]).fit().summary()


Variable names: A_B_C_D.

A = Num-> The number of bouts
A = Steps-> The number of steps

B = gr-> Greater than
B = ls-> Less than 

C = Med-> The median bout
C = 5/10 -> 5/10 minutes

D = 2-> From class 2 only
D = 3-> From class 3 only
D = 23-> From class 2 or 3

Example:
steps_ls_med_23 -> Number of steps in bouts less than the median bout duration in class 2 or 3.

def rle(arr, perc, step, to_min):
	w = percent_rule(arr, perc, step, to_min)
	we = [x[1]-x[0] for x in w]
	return _, we

def day_max(arr, check='gr', value=3):
	if arr.shape[0]!=10080:
		return np.array([0,0,0,0,0,0,0])
	else:
		re_arr = arr.reshape(7,1440)
		
		if check == 'gr':
			day_maxs = np.array([np.max(np.concatenate((rle(x>value)[1], [0]))) for x in re_arr])
		elif check=='eq':
			day_maxs = np.array([np.max(np.concatenate((rle(x==value)[1], [0]))) for x in re_arr])
		else:
			print('not valid input')
			return np.array([0,0,0,0,0,0,0])
		return day_maxs
		

ew = np.array([day_max(x[0][:, 1], value  = 2, perc, to_min) for x in s])		
days_above = np.array([np.sum(ew>=x,1) for x in np.arange(np.max(ew))])
to_plot = np.array([np.sum(days_above>=x, 1)/755 for x in np.arange(1,7)])
for i in np.arange(1,7):
	if i==1:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' day')
	else:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' days')
plt.legend()
plt.title('Class 2,3, 80%\nProportion of people with at least one bout of length X over N days')
plt.xlabel('Length of maximum bout (minutes)')
plt.ylabel('Proportion of people')
plt.show()

ew = np.array([day_max(x, check = 'eq', value  = 3) for x in encoded])		
days_above = np.array([np.sum(ew>=x,1) for x in np.arange(np.max(ew))])
to_plot = np.array([np.sum(days_above>=x, 1)/755 for x in np.arange(1,7)])
for i in np.arange(1,7):
	if i==1:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' day')
	else:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' days')
plt.legend()
plt.title('Class 3\nProportion of people with at least one bout of length X over N days')
plt.xlabel('Length of maximum bout (minutes)')
plt.ylabel('Proportion of people')
plt.show()

ew = np.array([day_max(x, check = 'gr', value  = 1) for x in encoded])		
days_above = np.array([np.sum(ew>=x,1) for x in np.arange(np.max(ew))])
to_plot = np.array([np.sum(days_above>=x, 1)/755 for x in np.arange(1,7)])
for i in np.arange(1,7):
	if i==1:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' day')
	else:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' days')
plt.legend()
plt.title('Class 2,3\nProportion of people with at least one bout of length X over N days')
plt.xlabel('Length of maximum bout (minutes)')
plt.ylabel('Proportion of people')
plt.show()




X = (1+sup_tab_num[np.arange(24)]).apply(np.log)
X = (X-X.mean())/X.std(ddof = 0)

clean_X = X[X.abs()<3.5]
clean_stan_table = stan_table[(stan_table.abs()<3.5)]

to_pred = ['M00_BestGrip', 'M00_SPPB_FMWSTime', 'M00_SPPB_CRTime','M00_MATSF_MATsfScore']

scores = np.zeros((4, 24))
for j in np.arange(24):
	for k, i in enumerate(to_pred):
		y = stan_table[[i]]
		X_in = clean_X[[clean_X.columns[j]]]

		y = y.apply(pd.to_numeric, errors='coerce')
		X_keep = np.sum(np.isnan(X_in),1).values!=0
		y_keep = np.isnan(y).values.flatten()
			
		to_keep = np.logical_not(np.logical_or(y_keep, X_keep))
		X_in = X_in[to_keep==1]
		y = y[to_keep==1]
			
		X_in = sm.add_constant(X_in)
			
		t1 = sm.OLS(y, X_in).fit()
		scores[k, j] = t1.rsquared
		
		t2 = sm.OLS(y, X_in[['const', clean_X.columns[j]]]).fit()
		
		
		
def rle(arr, perc, step, to_min):
	w = percent_rule(arr, perc, step, to_min)
	we = [x[1]-x[0] for x in w]
	return we

def day_max(arr, value, perc, to_min):
	if arr.shape[0]!=10080:
		return np.array([0,0,0,0,0,0,0])
	else:
		re_arr = arr.reshape(7,1440)
		
		if value == 3:
			day_maxs = np.array([np.max(np.concatenate((rle(x, perc, 50, to_min), [0]))) for x in re_arr])
		elif value==2:
			day_maxs = np.array([np.max(np.concatenate((rle(x, perc, 20, to_min), [0]))) for x in re_arr])
		else:
			print('not valid input')
			return np.array([0,0,0,0,0,0,0])
		return day_maxs
		

ew = np.array([day_max(x[0][:, 1], value  = 2, perc = 0.8, to_min = True) for x in s])		
days_above = np.array([np.sum(ew>=x,1) for x in np.arange(np.max(ew))])
to_plot = np.array([np.sum(days_above>=x, 1)/755 for x in np.arange(1,7)])

for i in np.arange(1,7):
	if i==1:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' day')
	else:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' days')
plt.legend()
plt.title('Class 2,3, 80%, Setting to minimum.\nProportion of people with at least one bout of length X over N days')
plt.xlabel('Length of maximum bout (minutes)')
plt.ylabel('Proportion of people')
plt.show()

ew = np.array([day_max(x[0][:, 1], value  = 3, perc = 0.8, to_min = True) for x in s])		
days_above = np.array([np.sum(ew>=x,1) for x in np.arange(np.max(ew))])
to_plot = np.array([np.sum(days_above>=x, 1)/755 for x in np.arange(1,7)])

for i in np.arange(1,7):
	if i==1:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' day')
	else:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' days')
plt.legend()
plt.title('Class 3, 80%, Setting to minimum.\nProportion of people with at least one bout of length X over N days')
plt.xlabel('Length of maximum bout (minutes)')
plt.ylabel('Proportion of people')
plt.show()

##
ew = np.array([day_max(x[0][:, 1], value  = 2, perc = 0.9, to_min = True) for x in s])		
days_above = np.array([np.sum(ew>=x,1) for x in np.arange(np.max(ew))])
to_plot = np.array([np.sum(days_above>=x, 1)/755 for x in np.arange(1,7)])

for i in np.arange(1,7):
	if i==1:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' day')
	else:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' days')
plt.legend()
plt.title('Class 2,3, 90%, Setting to minimum.\nProportion of people with at least one bout of length X over N days')
plt.xlabel('Length of maximum bout (minutes)')
plt.ylabel('Proportion of people')
plt.show()

ew = np.array([day_max(x[0][:, 1], value  = 3, perc = 0.9, to_min = True) for x in s])		
days_above = np.array([np.sum(ew>=x,1) for x in np.arange(np.max(ew))])
to_plot = np.array([np.sum(days_above>=x, 1)/755 for x in np.arange(1,7)])

for i in np.arange(1,7):
	if i==1:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' day')
	else:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' days')
plt.legend()
plt.title('Class 3, 90%, Setting to minimum.\nProportion of people with at least one bout of length X over N days')
plt.xlabel('Length of maximum bout (minutes)')
plt.ylabel('Proportion of people')
plt.show()
##

ew = np.array([day_max(x[0][:, 1], value  = 2, perc = 1.0, to_min = True) for x in s])		
days_above = np.array([np.sum(ew>=x,1) for x in np.arange(np.max(ew))])
to_plot = np.array([np.sum(days_above>=x, 1)/755 for x in np.arange(1,7)])

for i in np.arange(1,7):
	if i==1:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' day')
	else:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' days')
plt.legend()
plt.title('Class 2,3, 100%, Setting to minimum.\nProportion of people with at least one bout of length X over N days')
plt.xlabel('Length of maximum bout (minutes)')
plt.ylabel('Proportion of people')
plt.show()

ew = np.array([day_max(x[0][:, 1], value  = 3, perc = 1.0, to_min = True) for x in s])		
days_above = np.array([np.sum(ew>=x,1) for x in np.arange(np.max(ew))])
to_plot = np.array([np.sum(days_above>=x, 1)/755 for x in np.arange(1,7)])

for i in np.arange(1,7):
	if i==1:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' day')
	else:
		plt.plot(np.arange(np.max(ew)), to_plot[i-1], label = str(i)+' days')
plt.legend()
plt.title('Class 3, 100%, Setting to minimum.\nProportion of people with at least one bout of length X over N days')
plt.xlabel('Length of maximum bout (minutes)')
plt.ylabel('Proportion of people')
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import r2_score as r2
import warnings
import numpy
import itertools

def rle(bits):
  # make sure all runs of ones are well-bounded
  bounded = numpy.hstack(([0], bits, [0]))
  # get 1 at run starts and -1 at run ends
  difs = numpy.diff(bounded)
  run_starts, = numpy.where(difs > 0)
  run_ends, = numpy.where(difs < 0)
  return run_starts, run_ends - run_starts




# import file
tab = pd.read_csv('C:/Users/x/Documents/New folder/Merged_files_steps.csv', low_memory=False)
s = np.load('C:/Users/x/Documents/New folder/All_step data.npy', allow_pickle = True)
## Dependent varaibles-direct:
"""


Final Score
MAT score
PASE outside walking time

## Dependent variables to be computed
Z-scores of 4M walk
Z_scores of CR 
Z_scores grip
Average of 3 above z-scores
"""
# Generate FMWS score
tab['M00_SPPB_FMWSTime'] = tab[['M00_SPPB_FMWSAttempt2Time', 'M00_SPPB_FMWSAttempt1Time']].min(1)
# Outlier in FMWS set to nan


def pred_encode(arr1, arr2, tuple):
	
	encode_arr1 = encoder(arr1)
	encode_arr2 = encoder(arr2)
	
	if tuple[4]=='0':
		for i in np.arange(4):
			if tuple[i]=='1':
				encode_arr1[encode_arr2==i]=i
		return encode_arr1
	if tuple[4]=='1':
		for i in np.arange(4):
			if tuple[i]=='1':
				encode_arr2[encode_arr1==i]=i
		return encode_arr2
	
	
def encoder(arr1):
	ret = np.zeros(arr1.shape[0])
	for i in np.arange(arr1.shape[0]):
		if arr1[i]<10:
			ret[i] = 0
		elif arr1[i]<20:
			ret[i]=1
		elif arr1[i]<50:
			ret[i] = 2
		else:
			ret[i] = 3
	return ret
	
	
def shredder(arr_step, arr_class):
	time_in_0 = np.mean(arr_class==0)
	time_in_1 = np.mean(arr_class==1)
	time_in_2 = np.mean(arr_class==2)
	time_in_3 = np.mean(arr_class==3)
	step_in_1 = np.sum(arr_step[arr_class==1])
	step_in_2 = np.sum(arr_step[arr_class==2])
	step_in_3 = np.sum(arr_step[arr_class==3])
	return np.array([time_in_0, time_in_1, time_in_2, time_in_3, step_in_1, step_in_2, step_in_3])
	
def ids(full_id):
	return full_id.split('/')[-1].split('_')[0]
	
ui = [np.concatenate(([ids(x[2])], shredder(x[0][:, 1],encoder(x[0][:, 1])))) for x in s]

step_tab = pd.DataFrame(ui, columns = ['Small_ID', 'time_in_0', 'time_in_1', 'time_in_2', 'time_in_3', 'steps_in_1', 'steps_in_2','steps_in_3'])

sup_tab = pd.merge(tab, step_tab, left_on = 'ID', right_on = 'Small_ID')



x_val = ['total_steps', 'SCREENING_AgeAtRandomisation',
'SCREENING_Gender','M00_HW_BMI','M00_MedHist_NumPhysCondition','steps_in_3']

to_pred = 'M00_SPPB_FMWSTime'
data = sup_tab[x_val +['M00_SPPB_FMWSTime']].apply(pd.to_numeric, errors='coerce')
data['M00_SPPB_FMWSTime'][data['M00_SPPB_FMWSTime'].idxmax()] = np.nan
#data[data==0] = np.nan

data = data.dropna()
X1 = (data[x_val]+1).apply(np.log)
y1 = (data[to_pred]+1).apply(np.log)

model1 = sm.OLS(y1, sm.add_constant(X1)[sm.add_constant(X1).columns[:-1]]).fit()
print(model1.summary())


r = model1.resid

model2 = sm.OLS(r, sm.add_constant(X1['steps_in_3'])).fit()
print(model2.summary())





x = X1['steps_in_3']
plt.scatter(y1.values, r)
X_lin = np.arange(np.min(x), np.max(x), 0.01)
y_lin = model2.predict(sm.add_constant(X_lin))
plt.plot(X_lin, y_lin, color = 'red')
plt.xlabel('Logged total steps in class 3')
plt.ylabel('Residual of confounders model')
plt.xlim(2, 12)
plt.title('Regression showing the explanatory power of \n"steps in class 3" outside of other confounders')
plt.show()


























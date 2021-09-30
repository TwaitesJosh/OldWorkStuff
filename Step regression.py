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


# import file
tab = pd.read_csv('C:/Users/x/Documents/New folder/Merged_files_steps.csv', low_memory=False)
s = np.load('C:/Users/x/Documents/New folder/All_step data.npy', allow_pickle = True)
## Dependent varaibles-direct:
"""
Best grip
4M walk time
4M Score
CR time
CR score
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
		elif arr1[i]<70:
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

sup_tab1 = pd.merge(tab, step_tab, left_on = 'ID', right_on = 'Small_ID')


	
###
cont_variables = [
'M00_BestGrip',
'M00_SPPB_FMWSTime',
'M00_SPPB_CRTime',
'M00_MATSF_MATsfScore']

cont_table = sup_tab[cont_variables]
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

sup_tab_num = sup_tab.apply(pd.to_numeric, errors='coerce')+1
sup_tab_num['total_walk'] = sup_tab_num[['steps_in_2','steps_in_3']].sum(1)
sup_tab_num['not_fast_walk'] = sup_tab_num[['steps_in_2','steps_in_1']].sum(1)
sup_tab_num['total_steps'] = sup_tab_num[['steps_in_1','steps_in_2','steps_in_3']].sum(1)

sup_tab_num['total_walk_time'] = sup_tab_num[['time_in_2','time_in_3']].sum(1)
sup_tab_num['total_step_time'] = sup_tab_num[['time_in_1', 'time_in_2','time_in_3']].sum(1)

pX = sup_tab_num[['time_in_0', 'time_in_1', 'time_in_2', 'time_in_3', 'steps_in_1', 'steps_in_2', 'steps_in_3', 'total_walk', 'not_fast_walk','total_steps', 'total_walk_time', 'total_step_time']]
to_log = ['time_in_1', 'time_in_2', 'time_in_3', 'steps_in_1', 'steps_in_3']
for i in to_log:
	pX[i] = pX[i].apply(np.log)
	

clean_pX = pX[(((pX-pX.mean())/pX.std(ddof=0)).abs() <3.5)]
clean_stan_table = stan_table[(stan_table.abs() <3.5)]


for i in clean_stan_table.columns:
	y = clean_stan_table[i]
	X = clean_pX[['steps_in_3']]

	y = y.apply(pd.to_numeric, errors='coerce')
	to_keep = np.sum(np.isnan(X).values, 1)+np.isnan(y).values==0
	X = (X-X.mean())/X.std(ddof=0)
	X = X[to_keep]
	y = y[to_keep]
	
	X = sm.add_constant(X)
	
	t = sm.OLS(y, X).fit()
	print(t.summary())
	
	
	er = (X.values[:, 1] * t.params[1]) + t.params[0]
	plt.scatter(X.values[:, 1], y)
	plt.plot(X.values[:, 1], er, color = 'red')
	plt.title(i)
	plt.show()
	print(r2(y.values, er))
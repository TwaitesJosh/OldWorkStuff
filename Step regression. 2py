import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


# import file
tab = pd.read_csv('C:/Users/x/Documents/New folder/Merged_files_steps.csv', low_memory=False)

# Extract indep variables

indep_variables = ['M00_BestGrip',
'M00_SF6D_UtilityScore',
'M00_SPPB_FMWSScore',
'M00_SPPB_FTScore',
'M00_SPPB_BalanceScore',
'M00_SPPB_CRTime',
'M00_SPPB_CRScore',
'M00_SPPB_FinalScore',
'M00_SPPB_FinalScore_Z',
'M00_SPPB_FMWSBestTime']

indep_table = tab[indep_variables]
# convert all non numercics to nan's
indep_table = indep_table.apply(pd.to_numeric, errors='coerce')

# Extract dep variables
dep_variables = [
'M00_enmo',
'M00_boutsperperson',
'M00_totdurationperperson',
'M00_meanduration',
'compositecomplexity',
'compositedeterministicscore',
'volume',
'Steps_100',
'Steps_200']
dep_table = tab[dep_variables]

# convert all non-numercics to nan
dep_table = dep_table.apply(pd.to_numeric, errors='coerce')



dep_table.describe()
indep_table.describe()


corrmat = new_dep_table.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

new_dep = [
'M00_boutsperperson',
'M00_meanduration',
'Steps_100']
new_dep_table = dep_table[new_dep]
##

for i in dep_variables:
	plt.scatter(dep_table[i], indep_table['M00_BestGrip'])
	plt.xlabel(i)
	plt.ylabel('M00_BestGrip')
	plt.show()

##
create paired data table

Cont_variables = ['M00_BestGrip',
'M00_SF6D_UtilityScore',
'M00_SPPB_CRTime',
'M00_boutsperperson',
'M00_meanduration',
'Steps_100']
Cont_table = tab[Cont_variables]

# convert all non-numercics to nan
Cont_table = Cont_table.apply(pd.to_numeric, errors='coerce')

sns.set()
sns.pairplot(Cont_table, size = 2.5)
plt.show();

Disc_variables = ['M00_SPPB_FMWSScore',
'M00_SPPB_FTScore',
'M00_SPPB_BalanceScore',
'M00_SPPB_CRScore',
'M00_SPPB_FinalScore',
'M00_SPPB_FinalScore_Z',
'M00_boutsperperson',
'M00_meanduration',
'Steps_100']

Disc_table = tab[Disc_variables]
g = sns.PairGrid(Disc_table)
g.map_diag(sns.histplot)
g.map_offdiag(sns.boxplot)
plt.show();


for i in dep_variables:
	sns.distplot(dep_table[i], fit=norm);
	fig = plt.figure()
	plt.title(i)
	res = stats.probplot(dep_table[i], plot=plt)
	plt.show()



## Regressing for y
y = tab['M00_SPPB_FMWSScore']
X = tab[['M00_boutsperperson',
'M00_meanduration',
'Steps_100']]
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

to_keep = np.sum(np.isnan(X).values, 1)+np.isnan(y).values==0

X = X[to_keep]
y = y[to_keep]



# transform to classification
y = tab['M00_SPPB_FMWSScore']<2.5
X = tab[['M00_boutsperperson',
'M00_meanduration',
'Steps_100']]
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

to_keep = np.sum(np.isnan(X).values, 1)+np.isnan(y).values==0

X = X[to_keep]
y = y[to_keep]

#test if volume is better
y = tab['M00_SPPB_FMWSScore']
X1 = tab['volume']
X2 = tab['Steps_100']

X1 = X1.apply(pd.to_numeric, errors='coerce')
X2 = X2.apply(pd.to_numeric, errors='coerce')

y = y.apply(pd.to_numeric, errors='coerce')
y = y<1.2

to_keep1 = np.isnan(X1).values+np.isnan(y).values==0
to_keep2 = np.isnan(X2).values+np.isnan(y).values==0


X1 = X1[to_keep]
X2 = X2[to_keep]
y = y[to_keep]
lr = RC(class_weight = 'balanced')
score1 = ac(y, lr.fit(X1.values[:, None], y).predict(X1.values[:, None]))*100
score2 = ac(y, lr.fit(X2.values[:, None], y).predict(X2.values[:, None]))*100
print(score1-score2)


#### trying all the things
from sklearn.feature_selection import RFE
X = tab[tab.columns[np.concatenate((np.arange(1021,1059), np.array([1061, 1062])))]]
y = tab['M00_SPPB_FMWSScore']
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

to_keep = np.sum(np.isnan(X).values, 1)+np.isnan(y).values==0

X = X[to_keep]
y = y[to_keep]
lr = rc()
selector = RFE(lr, n_features_to_select = 4)
sl = selector.fit(X, y)
print(evs(y, sl.fit(X.values[:, :], y).predict(X.values[:,: ]))*100)
print(tab.columns[np.concatenate((np.arange(1021,1059), np.array([1061, 1062])))][sl.get_support()])

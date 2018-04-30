import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from scipy.stats import boxcox

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression


def best_corr(x, y, max_window = 52, method = 'pearson', diffs = (0, 1, 52)):
    windows = range(1, max_window, 1)
    quants = np.arange(.05, 1, .05)

    metrics = {
        'mean': lambda z: z.mean(),
        'min': lambda z: z.min(), 
        'max': lambda z: z.max(), 
        'sum': lambda z: z.sum(), 
        'var': lambda z: z.var(),
        'range': lambda z: z.apply(lambda a: a.max() - a.min()), 
        'prod': lambda z: z.apply(lambda a: a.prod()), 
        'first': lambda z: z.apply(lambda a: a[0]),
        'quantile': lambda z, q: z.quantile(q),
        'increase': lambda z: z.apply(lambda a: 1 if pd.Series(a).is_monotonic_increasing else 0),
        'decrease': lambda z: z.apply(lambda a: 1 if pd.Series(a).is_monotonic_decreasing else 0),
        'change': lambda z: z.apply(lambda a: a[len(a) - 1] - a[0])
    }
    
    best = {}
    best['corr'] = 0
    best['diff'] = 0
    
    for diff in diffs:
        if diff:
            series = x.diff(diff).copy()
        else:
            series = x.copy()
            
        for w in windows:
            for name, lamb in metrics.iteritems():
                if name == 'quantile':
                    best_quant = 0
                    for quant in quants:
                        cur_corr = lamb(series.rolling(w, min_periods = 1), quant).corr(y.tail(len(x) - w), method = method)
                        
                        if abs(cur_corr) > abs(new_corr):
                            new_corr = cur_corr
                            best_quant = quant
                else:
                    new_corr = lamb(series.rolling(w, min_periods = 1)).tail(len(x) - w).corr(y.tail(len(x) - w), method = method)
                    
                if abs(new_corr) > abs(best['corr']):
                    best['corr'] = new_corr
                    best['metric'] = name
                    best['window'] = w
                    best['diff'] = diff
                    
                    if name == 'quantile':
                        best['quant'] = best_quant
                    else:
                        best['quant'] = None
                    
    if best['diff']:
        if best['metric'] == 'quantile':
            best['method'] = lambda z: metrics[best['metric']](z.diff(best['diff']).rolling(best['window'], min_periods = 1), best['quant'])
        else:
            best['method'] = lambda z: metrics[best['metric']](z.diff(best['diff']).rolling(best['window'], min_periods = 1))
    else:
        if best['metric'] == 'quantile':
            best['method'] = lambda z: metrics[best['metric']](z.rolling(best['window'], min_periods = 1), best['quant'])
        else:
            best['method'] = lambda z: metrics[best['metric']](z.rolling(best['window'], min_periods = 1))
                
    return best

m1 = [31,28,31,30,31,30,31,31,30,31,30,31]
m2 = [31,29,31,30,31,30,31,31,30,31,30,31]

year0 = ['2011']
year5=['11']
year1 = ['2012','2013','2014']
year2 = ['2015','2016','2017']
year3 = ['12','13','14']
year4 = ['15','16','17']


##################dewp0#############
dewp0=[]
temp = pd.read_csv('Data/dewp2007-2012.csv')
for i in year0:

	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Date'].map(lambda x :x.split('/')[2])==i)]
	x=x['Avg'].tolist()
	dewp0 +=map(lambda x :sum(x)/7 ,np.reshape(x[0:364],(365/7,7)))


#########################dewp2##############################
dewp2=[]
temp = pd.read_csv('Data/dewp.csv')
for i in year2:

	x= temp[(temp['AT'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Date'].map(lambda x :x.split('/')[2])==i)]
	x=x['AVG'].tolist()
	dewp2 +=map(lambda x :sum(x)/7 ,np.reshape(x[0:364],(365/7,7)))


############################dewp1##############################
dewp1=[]
temp = pd.read_csv('Data/dewp2012-2014.csv')
for i in year1:

	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Date'].map(lambda x :x.split('/')[2])==i)]
	x=x['AVG'].tolist()
	dewp1 +=map(lambda x :sum(x)/7 ,np.reshape(x[0:364],(365/7,7)))

dewp = dewp0+dewp1+dewp2


############maxtemp 0#####################

maxtemp0 = []
temp = pd.read_csv('Data/maxtemp2007-2012.csv')
for i in year0:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('/')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 33.0 ,z)	
	maxtemp0+=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))


############maxtemp 1######################

maxtemp1 = []
temp = pd.read_csv('Data/maxtemp2012-2014.csv')
for i in year3:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','AVG'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 33.0 ,z)	
	maxtemp1 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))


####################maxtemp 2##########################
maxtemp2 = []
temp = pd.read_csv('Data/maxtemp.csv')
for i in year4:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','AVG'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 33.0 ,z)	
	maxtemp2 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))

maxtemp = maxtemp0+maxtemp1+maxtemp2


###################meanrh0##########################



meanrh0= []

temp = pd.read_csv('Data/mean-rh2007-2012.csv')
for i in year0:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('/')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 70 ,z)	
	meanrh0 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))


###################meanrh1##########################



meanrh1= []

temp = pd.read_csv('Data/mean-rh2012-2014.csv')
for i in year3:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','AVG'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 70 ,z)	
	meanrh1 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))

	############meanrh2############################

meanrh2= []

temp = pd.read_csv('Data/mean-rh.csv')
for i in year4:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','AVG'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 70 ,z)	
	meanrh2 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))

meanrh = meanrh0+meanrh1+meanrh2



################# meantemp0#############

meantemp0= []

temp = pd.read_csv('Data/meantemp2007-2012.csv')
for i in year0:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('/')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 33.0 ,z)	
	meantemp0 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))


################# meantemp1#############

meantemp1= []

temp = pd.read_csv('Data/meantemp2012-2014.csv')
for i in year3:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 33.0 ,z)	
	meantemp1 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))

	##########meantemp2####################

meantemp2= []

temp = pd.read_csv('Data/meantemp.csv')
for i in year4:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 33.0 ,z)	
	meantemp2 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))

meantemp = meantemp0+meantemp1+meantemp2

########mintemp0#######################

mintemp0= []

temp = pd.read_csv('Data/mintemp2007-2012.csv')
for i in year0:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('/')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 30.0 ,z)	
	mintemp0 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))


########mintemp#######################

mintemp1= []

temp = pd.read_csv('Data/mintemp2012-2014.csv')
for i in year3:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 30.0 ,z)	
	mintemp1 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))
#########Mintemp2##################

mintemp2= []

temp = pd.read_csv('Data/mintemp.csv')
for i in year4:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]

	z = map(lambda  x : float(x) if x!='-' else 30.0 ,z)	
	mintemp2 +=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))

mintemp = mintemp0+mintemp1+mintemp2


#######rain###################

rain0 = []

temp = pd.read_csv('Data/rain2007-2012.csv')
for i in year0:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('/')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]
	z = map(lambda x: 0 if x=='T' else x,z)
	z = map(lambda  x : float(x) if x!='-' else 0 ,z)	
	rain0+=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))

#######rain###################

rain1 = []

temp = pd.read_csv('Data/rain2012-2014.csv')
for i in year3:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','Avg'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]
	z = map(lambda x: 0 if x=='T' else x,z)
	z = map(lambda  x : float(x) if x!='-' else 0 ,z)	
	rain1+=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))
####rain2####################

rain2 = []

temp = pd.read_csv('Data/rain.csv')
for i in year4:
	x= temp[(temp['At'].map(lambda x :x.split('-')[0])=='455201')]
	x= x[(x['Month'].map(lambda x :x.split('-')[1])==i)]

	x=x.drop(['No','At','Month','Sum'],axis=1)
	y = x.values.tolist()
	z = []
	for i in range(12):
		for j in range(m1[i]):
			z+=[y[i][j]]
	z = map(lambda x: 0 if x=='T' else x,z)
	z = map(lambda  x : float(x) if x!='-' else 0 ,z)	
	rain2+=map(lambda x :sum(x)/7 ,np.reshape(z[0:364],(365/7,7)))

rain = rain0+rain1+rain2

all_data_x = pd.DataFrame({'dewp':dewp , ' maxtemp':maxtemp , 'mintemp':mintemp , 'meantemp':meantemp , 'meanrh':meanrh , 'rain':rain})
all_data_x.to_csv('all_455201.csv',index=False)
'''
all_data_y = pd.read_csv('Data/all_data_y.csv')
all_data_y = all_data_y.drop(['year','total_cases'],axis=1)
all_data = pd.concat([all_data_x,all_data_y],axis=1)

corrmat = all_data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
'''
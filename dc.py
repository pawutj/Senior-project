import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict

from xgboost import XGBRegressor
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

def epidemic(df):
    weekly_thresholds = df.groupby('week')['total_cases_adj'].quantile(.80).to_dict()
    ep = [1 if r['total_cases_adj'] > weekly_thresholds[r['week']] else 0 for i, r in df.iterrows()]
    
    return ep


all_data_x_1 = pd.read_csv('all_455201.csv',prefix='455201_')
all_data_x_2 = pd.read_csv('all_455203.csv',prefix='455203_')
all_data_x_3 = pd.read_csv('all_455301.csv',prefix='455301_')

all_data_x_1= all_data_x_1.add_prefix('455201_')
all_data_x_2= all_data_x_2.add_prefix('455203_')
all_data_x_3=all_data_x_3.add_prefix('455301_')

all_data_y = pd.read_csv('Data/all_data_y.csv')
all_data_y = all_data_y.drop(['total_cases'],axis=1)



list_year = [54,55,56,57,58,59,60]


#for i in list_year:
#	t = all_data_y[all_data_y['year']==i]
	#t=t.reset_index()
	#plt.plot(t['total_cases_adj'])

#plt.savefig('fig/1.jpg')
#plt.show()

#plt.plot(all_data_y['total_cases_adj'])
temp = pd.DataFrame({'total_cases_adj':all_data_y.rolling(3,min_periods=1).mean()['total_cases_adj']})
all_data_y_rolling = pd.concat([all_data_y.drop(['total_cases_adj'],axis=1),temp],axis=1)
#plt.plot(all_data_y_rolling['total_cases_adj'])
#plt.show()

all_data = pd.concat([all_data_x_1,all_data_x_2,all_data_x_3,all_data_y],axis=1)
'''
corrmat = all_data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
'''

all_data_rolling = pd.concat([all_data_x_1,all_data_x_2,all_data_x_3,all_data_y_rolling],axis=1)

'''
corrmat = all_data_rolling.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
'''


all_data_adj =all_data_rolling
col_name = all_data_rolling.columns[0:18]
'''
for i in col_name:
    
    best = best_corr(all_data_rolling[i],all_data_rolling['total_cases_adj'])
    print best
    all_data_adj[i]=best['method'](all_data_adj[i])
'''

#all_data_week = all_data['week']
#all_data_week = pd.get_dummies(all_data_week)

#lr = LinearRegression()
#lr.fit(all_data_week,all_data_rolling['total_cases_adj'])
#pred = lr.predict(all_data_week)
'''
plt.plot(pred)
plt.plot(all_data_rolling['total_cases_adj'])
plt.show()
'''
#z = all_data_rolling['total_cases_adj'].values - pred
#z = z.tolist()
#diff_pred  = pd.DataFrame({'deff_pred':z})
#all_data_rolling = pd.concat([all_data_rolling,diff_pred],axis=1)

'''
corrmat = all_data_rolling.corr()

plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
'''

train = all_data_rolling[(all_data_rolling['year']!=59) & (all_data_rolling['year']!=60) & (all_data_rolling['year']!=58)]
test = all_data_rolling[(all_data_rolling['year']==59) | (all_data_rolling['year']==60)]
test = test[test['total_cases_adj']<240]

print '#####################################################'
###########Start!!!!!!!!!!!!!#########################

#########Create Base line Model##################

print '########################1.Base Model##########################'

train_x = train.drop(['total_cases_adj'],axis=1)
train_y = train['total_cases_adj']
test_x = test.drop(['total_cases_adj'],axis=1)
test_y  = test['total_cases_adj']


lr = LinearRegression()

lr.fit(train_x,train_y)
pred = lr.predict(test_x)
print 'Linear Base Model Score'
print mean_absolute_error(pred,test_y)

#plt.plot(test_y.reset_index(drop=True))
#plt.plot(pred)
#plt.savefig('fig/2.jpg')
#plt.show()

xgb = XGBRegressor()
xgb.fit(train_x,train_y)
pred = xgb.predict(test_x)
print 'XGBR Base Model Score'
print mean_absolute_error(pred,test_y)

#plt.plot(test_y.reset_index(drop=True))
#plt.plot(pred)
#plt.savefig('fig/3.jpg')
#plt.show()
print '#################################################'
################################################

train = pd.concat([train,test],axis=0)
train = train.reset_index(drop=True)
###########Create trend#######################
print '######################2.Crete Trend###########################'
train_x= train['week']
train_x_week = pd.get_dummies(train_x)

test_x= test['week']
test_x_week = pd.get_dummies(test_x)

train_y = train['total_cases_adj']
test_y = test['total_cases_adj']

lr_trend = LinearRegression()
lr_trend.fit(train_x_week,train_y)
pred_trend = lr_trend.predict(test_x_week)

#print mean_absolute_error(pred_trend,test_y)

score_0 = cross_val_score(LinearRegression(), 
            train_x_week,
            train_y,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')

#plt.plot(test_y.reset_index(drop=True))
#plt.plot(pred_trend)
#plt.savefig('fig/4.jpg')
#plt.show()


###########predict from trend###########

train_trend = lr_trend.predict(train_x_week)
#plt.plot(train_trend)
#plt.plot(train_y)
#plt.savefig('fig/4.jpg')
#plt.show()
print 'Trend Model Score'
print mean_absolute_error(train_trend,train_y)
print '#############################################################'

##############predict with trend#############################

print '############################3.Predict With Trend############################'

train_y_diff = pd.DataFrame({'diff':(train_y-train_trend)})


train_withT = pd.concat([train,pd.DataFrame({'trend':train_trend})],axis=1)

corrmat = train_withT.corr()

#plt.subplots(figsize=(12,9))
#sns.heatmap(corrmat, vmax =1 ,vmin =-1, square=True,cmap="PiYG")
#plt.savefig('fig/5.jpg')
#plt.show()

score_1 = cross_val_score(LinearRegression(), 
            train.drop('total_cases_adj',axis=1),
            train_y_diff,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')

score_2 = cross_val_score(XGBRegressor(), 
            train.drop('total_cases_adj',axis=1),
            train_y_diff,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')

#score_3 = cross_val_score(LinearRegression(), 
 #           train.drop('total_cases_adj',axis=1),
  #          train_trend,
   #         cv = 5, 
    #        scoring = 'neg_mean_absolute_error')

#score_4 = cross_val_score(XGBRegressor(), 
 #           train.drop('total_cases_adj',axis=1),
  #          train_trend,
   #         cv = 5, 
    #        scoring = 'neg_mean_absolute_error')

score_5 = cross_val_score(LinearRegression(), 
            train.drop('total_cases_adj',axis=1),
            train['total_cases_adj'],
            cv = 10, 
            scoring = 'neg_mean_absolute_error')

score_6 = cross_val_score(XGBRegressor(), 
            train.drop('total_cases_adj',axis=1),
            train['total_cases_adj'],
            cv = 10, 
            scoring = 'neg_mean_absolute_error')

score_7 = cross_val_score(LinearRegression(), 
            train_withT.drop('total_cases_adj',axis=1),
            train['total_cases_adj'],
            cv = 10, 
            scoring = 'neg_mean_absolute_error')

score_8 = cross_val_score(XGBRegressor(), 
            train_withT.drop('total_cases_adj',axis=1),
            train['total_cases_adj'],
            cv = 10, 
            scoring = 'neg_mean_absolute_error')
print 'Linear Base Model'
print np.mean(score_5)
print 'XGBR Base Model'
print np.mean(score_6)

print 'Trend Score'
print np.mean(score_0)


#print np.mean(score_3)
#print np.mean(score_4)
print 'Linear  with Trend Param'
print np.mean(score_7)
print 'XGBR with Trend Param'
print np.mean(score_8)

print 'Linear to predict diff from Trend'
print np.mean(score_1)
print 'XGBR to predict diff from Trend'
print np.mean(score_2)

predict_1 = cross_val_predict(LinearRegression(), 
            train.drop('total_cases_adj',axis=1),
            train['total_cases_adj'],
            cv = 10, 
            )

predict_2 = cross_val_predict(LinearRegression(), 
            train.drop('total_cases_adj',axis=1),
            train['total_cases_adj'],
            cv = 10, 
            )

predict_3 = cross_val_predict(LinearRegression(), 
            train_withT.drop('total_cases_adj',axis=1),
            train['total_cases_adj'],
            cv = 10, 
            )

predict_4 = cross_val_predict(XGBRegressor(), 
            train_withT.drop('total_cases_adj',axis=1),
            train['total_cases_adj'],
            cv = 10, 
            )

predict_5 = cross_val_predict(LinearRegression(), 
            train.drop('total_cases_adj',axis=1),
            train_y_diff,
            cv = 10, 
            )

predict_6 = cross_val_predict(XGBRegressor(), 
            train.drop('total_cases_adj',axis=1),
            train_y_diff,
            cv = 10, 
            )


predict_5 = map(lambda x:x[0],predict_5)


#plt.plot(predict_1)
#plt.plot(train['total_cases_adj'])
#plt.savefig('fig/5.jpg')
#plt.show()

#plt.plot(predict_2)
#plt.plot(train['total_cases_adj'])
#plt.savefig('fig/6.jpg')
#plt.show()

#plt.plot(predict_3)
#plt.plot(train['total_cases_adj'])
#plt.savefig('fig/7.jpg')
#plt.show()

#plt.plot(predict_4)
#plt.plot(train['total_cases_adj'])
#plt.savefig('fig/8.jpg')
#plt.show()

#plt.plot(predict_5 + train_trend)
#plt.plot(train['total_cases_adj'])
#plt.savefig('fig/9.jpg')
#plt.show()

#plt.plot(predict_6 + train_trend)
#plt.plot(train['total_cases_adj'])
#plt.savefig('fig/10.jpg')
#plt.show()
print '################################################################################'
#######################################Corr ################################################
print '##########################4.Adjust Data from corr###################################'

train = pd.concat([train,pd.DataFrame({'trend':train_trend}),train_y_diff],axis=1)

#corrmat = train.corr()
#plt.subplots(figsize=(12,9))
#sns.heatmap(corrmat, vmax =1 ,vmin =-1, square=True,cmap="PiYG")
#plt.savefig('fig/11.jpg')
#plt.show()

train_forcorr = train.drop(['week','year','total_cases_adj','trend'],axis=1)
'''
train_adj = train

for r in train_forcorr.columns[:-1]:
    best = best_corr(train_forcorr[r],train_forcorr['diff'])
    print best
    train_adj[r] = best['method'](train_forcorr[r])
'''

train_adj = pd.read_csv('train_adj.csv')

#corrmat = train_adj.corr()
#plt.subplots(figsize=(12,9))
#sns.heatmap(corrmat, vmax =1 ,vmin =-1, square=True,cmap="PiYG")
#plt.savefig('fig/15.jpg')
#plt.show()

train_trend = train_adj.iloc[52:,:]['trend']
train_true = train_adj.iloc[52:,:]['total_cases_adj']
train_adj_x = train_adj.iloc[52:,:].drop(['week','year','total_cases_adj','trend','diff'],axis=1)
train_adj_y = train_adj.iloc[52:,:]['diff']
train_adj_x = train_adj_x.drop(['455201_meantemp'],axis=1)
score_1 = cross_val_score(LinearRegression(), 
            train_adj_x,
            train_adj_y,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')

score_2 = cross_val_score(XGBRegressor(), 
            train_adj_x,
            train_adj_y,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')

print 'Linear with diff + corr adjust'
print np.mean(score_1)
print 'XGBR with diff + corr adjust'
print np.mean(score_2)

predict_2 = cross_val_predict(XGBRegressor(),
                train_adj_x,
                train_adj_y,
                cv=10
                )

#plt.plot(train_true)
#plt.plot(train_trend+predict_2)
#plt.savefig('fig/12.jpg')
#plt.show()
print '##########################################################'

#################feature selection##############################
print '#################5.Feature Selection########################'
temp = []
temp2=[]

print 'Featrue Selection#1'

for r in train_adj_x.columns:
    score_2 = cross_val_score(XGBRegressor(), 
            train_adj_x.drop([r],axis=1),
            train_adj_y,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')
    temp+=[np.mean(score_2)]
    temp2+=[r]

print zip(temp,temp2)
print ''

train_adj_x_s = train_adj_x.drop(['455201_mintemp','455201_rain','455301_meanrh'],axis=1)

score_2 = cross_val_score(XGBRegressor(), 
            train_adj_x_s,
            train_adj_y,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')
print 'after Featrue Selection#1'
print np.mean(score_2)
print ''
temp = []
temp2=[]
print 'Feature Selecion#2'
for r in train_adj_x_s.columns:
    score_2 = cross_val_score(XGBRegressor(), 
            train_adj_x_s.drop([r],axis=1),
            train_adj_y,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')
    temp+=[np.mean(score_2)]
    temp2+=[r]
print zip(temp,temp2)
print ''

train_adj_x_s = train_adj_x_s.drop(['455203_meantemp','455201_dewp'],axis=1)

score_2 = cross_val_score(XGBRegressor(), 
            train_adj_x_s,
            train_adj_y,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')
print 'after Featrue Selection#2'
print np.mean(score_2)
print ''
temp = []
temp2=[]
print 'Feature Selection#3'
for r in train_adj_x_s.columns:
    score_2 = cross_val_score(XGBRegressor(), 
            train_adj_x_s.drop([r],axis=1),
            train_adj_y,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')
    temp+=[np.mean(score_2)]
    temp2+=[r]

print zip(temp,temp2)
print ''

#predict_2 = cross_val_predict(XGBRegressor(),
#                train_adj_x_s,
#                train_adj_y,
#                cv=10
#                )

#plt.plot(train_true)
#plt.plot(train_trend+predict_2)
#plt.show()
print '############################################################'

'''
for i in range(2,12):
    param ={
        'max_depth' : i
    }
    score = cross_val_score(XGBRegressor(**param),
                train_adj_x_s,
                train_adj_y,
                cv=10,
                scoring = 'neg_mean_absolute_error')
    print str(i) 
    print np.mean(score)
'''
#################ModelSection###################################
print '######################6.Model Selection########################'

score = cross_val_score(XGBRegressor(), 
            train_adj_x_s,
            train_adj_y,
            cv = 10, 
            scoring = 'neg_mean_absolute_error')
print 'XGBR'
print np.mean(score)

score = cross_val_score(LinearRegression(),
                train_adj_x_s,
                train_adj_y,
                cv=10,
                scoring = 'neg_mean_absolute_error')
print 'Linear'
print np.mean(score)

score = cross_val_score(Ridge(),
                train_adj_x_s,
                train_adj_y,
                cv=10,
                scoring = 'neg_mean_absolute_error')

print 'Ridge'
print np.mean(score)

score = cross_val_score(ElasticNet(),
                train_adj_x_s,
                train_adj_y,
                cv=10,
                scoring = 'neg_mean_absolute_error')

print 'ElasticNet'
print np.mean(score)

score = cross_val_score(SVR(C = 1000, gamma = .001),
                train_adj_x_s,
                train_adj_y,
                cv=10,
                scoring = 'neg_mean_absolute_error')

print 'SVR'
print np.mean(score)

score = cross_val_score(GradientBoostingRegressor(n_estimators = 1000, max_depth = 10),
                train_adj_x_s,
                train_adj_y,
                cv=10,
                scoring = 'neg_mean_absolute_error')

print 'GBR'
print np.mean(score)

score = cross_val_score(RandomForestRegressor(n_estimators = 500),
                train_adj_x_s,
                train_adj_y,
                cv=10,
                scoring = 'neg_mean_absolute_error')

print 'RFR'
print np.mean(score)

'''
for i in range(1,10):
    score = cross_val_score(RandomForestRegressor(n_estimators = i*50),
                train_adj_x_s,
                train_adj_y,
                cv=10,
                scoring = 'neg_mean_absolute_error')

    print i
    print np.mean(score)
'''
print '#######################################################'
#####em###################

print '#################7.ensemble model########################'
predict_1 = cross_val_predict(ElasticNet(),
                train_adj_x_s,
                train_adj_y,
                cv=10
                )

predict_2 = cross_val_predict(XGBRegressor(),
                train_adj_x_s,
                train_adj_y,
                cv=10
                )

en_predict = (predict_1+predict_2)/2
print 'ensemble xgbR + elasticR'
print mean_absolute_error((predict_1+predict_2)/2 , train_adj_y)

train_trend = train_trend.reset_index(drop=True)
train_true = train_true.reset_index(drop=True)
#plt.plot(train_true)
#plt.plot(predict_1+train_trend)
#plt.plot(predict_2+train_trend)
#plt.savefig('fig/13.jpg')
#plt.show()

#plt.plot(train_true)
#plt.plot(train_trend+en_predict)
#plt.savefig('fig/14.jpg')
#plt.show()

print '####################################################'
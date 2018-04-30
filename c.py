import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
x = pd.read_csv('Data/meantemp.csv',header=None)
y = x.iloc[0:12,:]
y = y.iloc[:,3:]
y = y.drop([34],axis=1)
z=[]
for i in y.values.reshape(31*12):
	if i != '-':
		z+=[i]
z = np.asarray(z[8:])
z= z.reshape(51,7).astype(float)
meantemp = z.mean(axis=1)



x = pd.read_csv('Data/mintemp.csv',header=None)
y = x.iloc[0:12,:]
y = y.iloc[:,3:]
for i in range(3,28):
	y[i]=y[i].apply(lambda s: s if s!='-' else 25)
y = y.drop([34],axis=1)
z=[]
for i in y.values.reshape(31*12):
	if i != '-':
		z+=[i]
z = np.asarray(z[8:])

z= z.reshape(51,7).astype(float)
mintemp = z.mean(axis=1)



x = pd.read_csv('Data/maxtemp.csv',header=None)
y = x.iloc[0:12,:]
y = y.iloc[:,3:]

for i in range(3,28):
	y[i]=y[i].apply(lambda s: s if s!='-' else 25)

y = y.drop([34],axis=1)
z=[]
for i in y.values.reshape(31*12):
	if i != '-':
		z+=[i]
z = np.asarray(z[8:])

z= z.reshape(51,7).astype(float)
maxtemp = z.mean(axis=1)


x = pd.read_csv('Data/mean-rh.csv',header=None)
y = x.iloc[0:12,:]
y = y.iloc[:,3:]

for i in range(3,28):
	y[i]=y[i].apply(lambda s: s if s!='-' else 70)

y = y.drop([34],axis=1)
z=[]
for i in y.values.reshape(31*12):
	if i != '-':
		z+=[i]
z = np.asarray(z[8:])

z= z.reshape(51,7).astype(float)
meanrh = z.mean(axis=1)




x = pd.read_csv('Data/rain.csv',header=None)
y = x.iloc[0:12,:]
y = y.iloc[:,3:]

for i in range(3,31):
	y[i]=y[i].apply(lambda s: s if (s!='-' and s!='T') else 0)

y.loc[0,[31,32]]=0
y.loc[2,[32,33]]=0
y.loc[3,[31]]=0
y.loc[4,[31,32,33]]=0
y.loc[5,[31,32]]=0
y.loc[7,[31,32]]=0
y.loc[9,31]=0
y.loc[10,[31,32]]=0
y.loc[11,[31,32,33]]=0

y = y.drop([34],axis=1)

z=[]
for i in y.values.reshape(31*12):
	if i != '-':
		z+=[i]
z = np.asarray(z[8:])

z= z.reshape(51,7).astype(float)
rain = z.sum(axis=1)

x = pd.read_csv('Data/label.csv')
x = x.values.tolist()[0]
df = pd.DataFrame({'x':x})
df.loc[41,'x']=11950
t = df.loc[:,'x'].values.tolist()
r=[]
for i in range(0,len(t)-1):
	r.append(t[i+1]-t[i])
	if r[i] <0:
		r[i]=0
y=pd.DataFrame({'r':r})
x = pd.DataFrame({'mintemp':mintemp,'maxtemp':maxtemp,'meantemp':meantemp,'meanrh':meanrh,'rain':rain})
X_train,X_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=3131)
 

xgb=XGBRegressor()
xgb.fit(X_train,y_train)
pred = pd.DataFrame({'pred':xgb.predict(X_test)})
y_test=y_test.reset_index()
z = pd.concat((pred,y_test),axis=1)

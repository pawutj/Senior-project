import pandas as pd


m = [31,28,31,30,31,30,31,31,30,31,30,31]

x = pd.read_csv('Data/rain.csv',header=None)
y=x.drop([0,1,2],axis=1)

y=y.iloc[0:12,]	


for i in range(0,12):
	y.loc[i,:] = y.loc[i,:].map(lambda x: x if x!='-' else 0)

rain =[]

for i in range(0,12):
	for j in range(0,m[i]):
		rain+=[y.iloc[i,j]]


x = pd.read_csv('Data/dewp.csv',header=None)
y = x.drop(range(0,11),axis=1)
dewp = y.iloc[0:365,:][11].tolist()


x = pd.read_csv('Data/maxtemp.csv',header=None)
y = x.drop([0,1,2],axis=1)

for i in range(12):
	y.iloc[i,:] = y.iloc[i,:].map(lambda x : x if x!='-' else y.iloc[i,31])
maxtemp = []
for i in range(12):
	for j in range(0,m[i]):
		maxtemp+=[y.iloc[i,j]]


x = pd.read_csv('Data/meantemp.csv',header=None)
y = x.drop([0,1,2],axis=1)

for i in range(12):
	y.iloc[i,:] = y.iloc[i,:].map(lambda x : x if x!='-' else y.iloc[i,31])
meantemp = []
for i in range(12):
	for j in range(0,m[i]):
		meantemp+=[y.iloc[i,j]]

x = pd.read_csv('Data/mintemp.csv',header=None)
y = x.drop([0,1,2],axis=1)

for i in range(12):
	y.iloc[i,:] = y.iloc[i,:].map(lambda x : x if x!='-' else y.iloc[i,31])
mintemp = []
for i in range(12):
	for j in range(0,m[i]):
		mintemp+=[y.iloc[i,j]]

x = pd.read_csv('Data/mean-rh.csv',header=None)
y = x.drop([0,1,2],axis=1)

for i in range(12):
	y.iloc[i,:] = y.iloc[i,:].map(lambda x : x if x!='-' else y.iloc[i,31])
mean_rh = []
for i in range(12):
	for j in range(0,m[i]):
		mean_rh+=[y.iloc[i,j]]

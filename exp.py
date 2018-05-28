import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

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

print all_data.drop(['year','week'],axis=1).describe()

'''
[u'455201_ maxtemp', u'455201_dewp', u'455201_meanrh',
       u'455201_meantemp', u'455201_mintemp', u'455201_rain',
       u'455203_ maxtemp', u'455203_dewp', u'455203_meanrh',
       u'455203_meantemp', u'455203_mintemp', u'455203_rain',
       u'455301_ maxtemp', u'455301_dewp', u'455301_meanrh',
       u'455301_meantemp', u'455301_mintemp', u'455301_rain', u'week', u'year',
       u'total_cases_adj']
'''

'''
temp = pd.concat([all_data['455201_meantemp'],all_data['455203_meantemp'],all_data['455301_meantemp']],axis=1)
plt.figure
temp.plot.hist(alpha=0.5)
#plt.show()
plt.savefig('fig/e1.jpg')


temp = pd.concat([all_data['455201_ maxtemp'],all_data['455203_ maxtemp'],all_data['455301_ maxtemp']],axis=1)
plt.figure
temp.plot.hist(alpha=0.5)
#plt.show()
plt.savefig('fig/e2.jpg')

temp = pd.concat([all_data['455201_mintemp'],all_data['455203_mintemp'],all_data['455301_mintemp']],axis=1)
plt.figure
temp.plot.hist(alpha=0.5)
#plt.show()
plt.savefig('fig/e3.jpg')

temp = pd.concat([all_data['455201_rain'],all_data['455203_rain'],all_data['455301_rain']],axis=1)
plt.figure
temp.plot.hist(alpha=0.5)
#plt.show()
plt.savefig('fig/e4.jpg')


temp = pd.concat([all_data['455201_meanrh'],all_data['455203_meanrh'],all_data['455301_meanrh']],axis=1)
plt.figure
temp.plot.hist(alpha=0.5)
#plt.show()
plt.savefig('fig/e5.jpg')

temp = pd.concat([all_data['455201_dewp'],all_data['455203_dewp'],all_data['455301_dewp']],axis=1)
plt.figure
temp.plot.hist(alpha=0.5)
#plt.show()
plt.savefig('fig/e6.jpg')
'''
	

temp = pd.concat([all_data['455201_meantemp'],all_data['455201_ maxtemp'],all_data['455201_mintemp'],all_data['455201_rain'],all_data['455201_meanrh'],all_data['455201_dewp'],all_data['total_cases_adj']],axis=1)
temp.corr()['total_cases_adj'][:-1].sort_values().plot.barh()
#plt.show()
plt.savefig('fig/e7.jpg')

temp = pd.concat([all_data['455203_meantemp'],all_data['455203_ maxtemp'],all_data['455203_mintemp'],all_data['455203_rain'],all_data['455203_meanrh'],all_data['455203_dewp'],all_data['total_cases_adj']],axis=1)
temp.corr()['total_cases_adj'][:-1].sort_values().plot.barh()
#plt.show()
plt.savefig('fig/e8.jpg')


temp = pd.concat([all_data['455301_meantemp'],all_data['455301_ maxtemp'],all_data['455301_mintemp'],all_data['455301_rain'],all_data['455301_meanrh'],all_data['455301_dewp'],all_data['total_cases_adj']],axis=1)
temp.corr()['total_cases_adj'][:-1].sort_values().plot.barh()
#plt.show()
plt.savefig('fig/e9.jpg')
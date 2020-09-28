import pyodbc
import pandas as pd
import numpy as np
from sklearn import linear_model
#%%
server = 'wasteless.database.windows.net' 
database = 'Wasteless' 
username = 'wasteless' 
password = 'oDrv3gLATxuO9i5382OC' 
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()


df = pd.read_sql_query('select x.DateSid, ProductionWasteKg, LineWasteKg, PlateWasteKg, WasteTotalKg,MealCount, SpecialMealCount, MealTotal, Menu, 1 LocationSID from dw.dimDate x left join [DW].[FacWasteless] y on x.datesid = y.datesid and y.LocationSID = 1 left join DW.FacMenu z on z.DateSID = x.DateSID and z.LocationSID = 1 where x.Date <= dateadd(day,5,getdate()) and x.Date >= convert(datetime, convert(char(8),20171127)) and x.WorkDay = 1 order by datesid asc',cnxn)    
#%%
df['Date'] = pd.to_datetime(df['DateSid'], format='%Y%m%d')
today = pd.to_datetime('2020-02-24')
#%%
#print(type(df['Menu'][0]))

#%%
df.dropna(subset=['Menu'], inplace=True)
df['MenuClass'] = df.apply(lambda row: row['Menu'].split(',', 1)[0].split('/', 1)[0].split(' ', 1)[0], axis=1)
df.replace(to_replace=['\nTalon', 'Hedelmäinen', 'Makkara-', 'Kala(kasvis)keitto', 'Ohra-riisipuuro', 'Suikalelihakastike', 'Kappalekala', 'Broileripatukka'],
           value=['Talon', 'Broilerikastike', 'Makkarakeitto', 'Kalakeitto', 'Riisi-ohrapuuro', 'Lihakastike', 'Kalaleike', 'Broilerinugetti'], inplace=True)

df = pd.concat([df, pd.get_dummies(df['MenuClass'], prefix='MenuCode')], axis=1)
menucols = [col for col in df.columns if 'MenuCode' in col]
#%%
for i in range(1,6):
    df['MealTotalLag'+str(i)] = df['MealTotal'].shift(i)
df['MealTotalMA'] = df.apply(lambda row: np.mean([row['MealTotalLag1'],row['MealTotalLag2'],row['MealTotalLag3'],row['MealTotalLag4'],row['MealTotalLag5']]), axis=1)

#%%
dftrain = df.dropna(subset=['MealTotal'])
dftrain.dropna(subset=['MealTotalMA'], inplace=True)
dftrain = dftrain[dftrain['MealTotal'] < 600]

#%%
Xmenu = dftrain[menucols].values
Xother = dftrain[['MealTotalMA']].values
X = np.concatenate((Xmenu, Xother), axis=1)
y = np.array(dftrain['MealTotal'])

modelMeal = linear_model.LinearRegression()
modelMeal.fit(X,y)
dftrain['PredictedMealTotal'] = modelMeal.predict(X)
#%%
dftrainwaste = df.dropna(subset=['WasteTotalKg'])
Xmenu = dftrainwaste[menucols].values
#Xother = dftrainwaste[['MealTotalMA']].values
#X = np.concatenate((Xmenu, Xother), axis=1)
X = Xmenu
y = np.array(dftrainwaste['WasteTotalKg'])

modelWaste = linear_model.LinearRegression()
modelWaste.fit(X,y)
dftrainwaste['PredictedWasteTotalKg'] = modelWaste.predict(X)

#%%
dftest = df.loc[(df['Date'] >= today) & (df['Date'] < today + pd.DateOffset(days=8))]
dftest['MealTotalMA'].fillna(method='ffill', inplace=True)
Xtestmenu = dftest[menucols].values
Xtestother = dftest[['MealTotalMA']].values
Xtest = np.concatenate((Xtestmenu, Xtestother), axis=1)
dftest['PredictedMealTotal'] = modelMeal.predict(Xtest)
Xtest = Xtestmenu
dftest['PredictedWasteTotalKg'] = modelWaste.predict(Xtest)
#%%


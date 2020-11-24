import pyodbc
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import preprocessing
sns.set_style('darkgrid')

pyodbc.lowercase = False

#%%
server = 'wasteless.database.windows.net' 
database = 'Wasteless' 
username = 'wasteless' 
password = 'oDrv3gLATxuO9i5382OC' 
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

df = pd.read_sql_query('select x.DateSID, ProductionWasteKg, LineWasteKg, PlateWasteKg, WasteTotalKg,MealCount, SpecialMealCount, MealTotal, Menu, PELessonBeforeLunchTurnout, HouseholdLessonBeforeLunchTurnout, StudentCountTotal, Cloudiness, Temperature, WindSpeed, 1 LocationSID from dw.dimDate x left join [DW].[FacWasteless] y on x.datesid = y.datesid and y.LocationSID = 1 left join DW.FacMenu z on z.DateSID = x.DateSID and z.LocationSID = 1 left join DW.FacTimeTableExtra v on v.DateSID = x.DateSID and v.LocationSID = 1 left join DW.FacStudentCountPerDay w on w.DateSID=x.DateSID and w.LocationSID=1 left join DW.FacWeatherInfo q on q.DateSID=x.DateSID and q.LocationSID=1 where x.Date <= dateadd(day,8,getdate()) and x.Date >= convert(datetime, convert(char(8),20171127)) and x.WorkDay = 1 order by datesid asc',cnxn)    
dfjs = pd.read_sql_query('select x.DateSID, ProductionWasteKg, LineWasteKg, PlateWasteKg, WasteTotalKg,MealCount, SpecialMealCount, MealTotal, Menu, PELessonBeforeLunchTurnout, HouseholdLessonBeforeLunchTurnout, StudentCountTotal, Cloudiness, Temperature, WindSpeed, 2 LocationSID from dw.dimDate x left join [DW].[FacWasteless] y on x.datesid = y.datesid and y.LocationSID = 2 left join DW.FacMenu z on z.DateSID = x.DateSID and z.LocationSID = 2 left join DW.FacTimeTableExtra v on v.DateSID = x.DateSID and v.LocationSID = 2 left join DW.FacStudentCountPerDay w on w.DateSID=x.DateSID and w.LocationSID=2 left join DW.FacWeatherInfo q on q.DateSID=x.DateSID and q.LocationSID=2 where x.Date <= dateadd(day,8,getdate()) and x.Date >= convert(datetime, convert(char(8),20171127)) and x.WorkDay = 1 order by datesid asc',cnxn)    


#df = pd.read_sql_query('select x.DateSID, ProductionWasteKg, LineWasteKg, PlateWasteKg, WasteTotalKg,MealCount, SpecialMealCount, MealTotal, Menu, PELessonBeforeLunchTurnout, HouseholdLessonBeforeLunchTurnout, 1 LocationSID from dw.dimDate x left join [DW].[FacWasteless] y on x.datesid = y.datesid and y.LocationSID = 1 left join DW.FacMenu z on z.DateSID = x.DateSID and z.LocationSID = 1 left join DW.FacTimeTableExtra v on v.DateSID = x.DateSID where x.Date <= dateadd(day,5,getdate()) and x.Date >= convert(datetime, convert(char(8),20171127)) and x.WorkDay = 1 order by datesid asc',cnxn)    
dfevents = pd.read_sql_query('select DateSID, EventDesc, PackedLunchCount, LocationSID from DW.FacEventCalendar', cnxn)
#dfstudents = pd.read_sql_query('select DateSID, StudentCountTotal from DW.FacStudentCountPerDay', cnxn)
#dfweather = pd.read_sql_query('select DateSID, Cloudiness, Temperature, WindSpeed from DW.FacWeatherInfo', cnxn)
df = df.append(dfjs)

#%%
df['Date'] = pd.to_datetime(df['DateSID'], format='%Y%m%d')
dfevents['Date'] = pd.to_datetime(dfevents['DateSID'], format='%Y%m%d')
#dfstudents['Date'] = pd.to_datetime(dfstudents['DateSID'], format='%Y%m%d')
#dfweather['Date'] = pd.to_datetime(dfweather['DateSID'], format='%Y%m%d')

today = pd.to_datetime('2020-9-2')

#%%
#dfevents['PackedLunchCount'].fillna(0, inplace=True)
df['PackedLunchCount'] = df.apply(lambda row: np.max(dfevents['PackedLunchCount'].loc[(dfevents['Date']==row['Date']) & (dfevents['LocationSID']==row['LocationSID'])]), axis=1)
df['PackedLunchCount'].fillna(0, inplace=True)
df['Event'] = df.apply(lambda row: '-'.join(dfevents['EventDesc'].loc[(dfevents['Date']==row['Date']) & (dfevents['LocationSID']==row['LocationSID'])]), axis=1)
#df['EventDesc'].fillna('-', inplace=True)
df['TET'] = df.apply(lambda row: 1 if 'TET' in row['Event'] else 0, axis=1)
df['Leirikoulu'] = df.apply(lambda row: 1 if 'leirikoulu' in row['Event'] else 0, axis=1)
#df['StudentCountTotal'] = df.apply(lambda row: np.max(dfstudents['StudentCountTotal'].loc[dfstudents['Date']==row['Date']]), axis=1)
df['StudentCountTotal'].fillna(np.mean(df['StudentCountTotal'].dropna()), inplace=True)
df['HouseholdLessonBeforeLunchTurnout'].fillna(np.mean(df['HouseholdLessonBeforeLunchTurnout'].dropna()), inplace=True)
df['PELessonBeforeLunchTurnout'].fillna(np.mean(df['PELessonBeforeLunchTurnout'].dropna()), inplace=True)
#df['Cloudiness'] = df.apply(lambda row: np.max(dfweather['Cloudiness'].loc[dfweather['Date']==row['Date']]), axis=1)
df['Cloudiness'].fillna(method='ffill', inplace=True)
#df['Temperature'] = df.apply(lambda row: np.max(dfweather['Temperature'].loc[dfweather['Date']==row['Date']]), axis=1)
df['Temperature'].fillna(method='ffill', inplace=True)
#df['WindSpeed'] = df.apply(lambda row: np.max(dfweather['WindSpeed'].loc[dfweather['Date']==row['Date']]), axis=1)
df['WindSpeed'].fillna(method='ffill', inplace=True)
df['WindSpeed'] = df.apply(lambda row: float(row['WindSpeed'].replace(',','.')), axis=1)


#%%
df.dropna(subset=['Menu'], inplace=True)
df['Menu'] = df.apply(lambda row: row['Menu'] if row['Menu'][0]!=' ' else row['Menu'][1:], axis=1)
df['MenuClass'] = df.apply(lambda row: row['Menu'].split(',', 1)[0].split('/', 1)[0].split(' ', 1)[0], axis=1)
df.replace(to_replace=['\nTalon', 'Hedelmäinen', 'Makkara-', 'Kala(kasvis)keitto', 'Ohra-riisipuuro', 'Suikalelihakastike', 'Kappalekala', 'Broileripatukka', 'Kiusaus'+chr(160)+'(kebab', 'Kebab-', 'Kalakasviskeitto', 'Kesäkeitto', 'Kalamurekepihvi', 'Riisi-ohrapuuro',
                       'Vaalea', 'Ohrapuuro', 'Lohikeitto', 'Liha-makaronipata', 'Broilerihoukutus', 'Ranskalainen', 'Juustoinen', 'Tomaattinen', 'Lohikiusaus', 'Kalkkunahöystö', 'Kalakepukka', 'Juuresvoipapupyörykkä', 'Broileri-kasviskastike', 'Kana-kookoskastike', 'Jauheliha-juustoperunavuoka', 'Porkkanapyörykkä', 'Naudan', 'Broileripyo'+chr(776)+'rykka'+chr(776), 'Kanapasta', 'Pinaattiohukkaat', 'Italianpata'],
           value=['Talon', 'Broilerikastike', 'Makkarakeitto', 'Kalakeitto', 'Puuro', 'Lihakastike', 'Kalaleike', 'Broilerinugetti', 'Kiusaus', 'Kiusaus', 'Kalakeitto', 'Kasviskeitto', 'Kalamureke', 'Puuro',
                  'Kasviskastike', 'Puuro', 'Kalakeitto', 'Makaronipata', 'Broilerikastike', 'Kalaleike', 'Broilerikeitto', 'Jauhelihavuoka', 'Kiusaus', 'Kalkkunakastike', 'Kalapuikot', 'Kasvispyörykät', 'Broilerikastike', 'Broilerikastike', 'Jauhelihavuoka', 'Kasvispyörykät', 'Jauhelihapihvi', 'Broileripyörykkä', 'Broileripasta', 'Pinaattiohukaiset', 'Makaronipata'], inplace=True)

df = pd.concat([df, pd.get_dummies(df['MenuClass'], prefix='MenuCode')], axis=1)
menucols = [col for col in df.columns if 'MenuCode' in col]

menuitems = df['MenuClass'].unique().tolist()

#%%
for i in range(1,6):
    df['MealCountLag'+str(i)] = df['MealCount'].shift(i)
df['MealCountMA'] = df.apply(lambda row: np.mean([row['MealCountLag1'],row['MealCountLag2'],row['MealCountLag3'],row['MealCountLag4'],row['MealCountLag5']]), axis=1)
df['MealCountMA'].fillna(method='ffill', inplace=True)

#%%
for i in range(1,15):
    df['TemperatureLag'+str(i)] = df['Temperature'].shift(i)
df['TemperatureMA'] = df.apply(lambda row: np.mean([row['TemperatureLag1'],row['TemperatureLag2'],row['TemperatureLag3'],row['TemperatureLag4'],row['TemperatureLag5'],row['TemperatureLag6'],row['TemperatureLag7'],row['TemperatureLag5'],row['TemperatureLag8'],row['TemperatureLag9'],row['TemperatureLag10'],row['TemperatureLag11'],row['TemperatureLag12'],row['TemperatureLag13'], row['TemperatureLag14'],]), axis=1)
df['TemperatureMA'].fillna(method='ffill', inplace=True)
df['TemperatureMA'].fillna(method='bfill', inplace=True)
df['RelativeTemperature'] = df.apply(lambda row: row['Temperature']/row['TemperatureMA'], axis=1)

#%%
dftrain = df.dropna(subset=['MealCount'])
dftrain.dropna(subset=['MealCountMA'], inplace=True)
dftrain = dftrain[dftrain['MealCount'] < 600]
dftrain = dftrain[dftrain['MealCount'] > 10]

#%%
othercols = ['LocationSID', 'StudentCountTotal', 'MealCountMA', 'TET', 'Leirikoulu', 'PackedLunchCount', 'HouseholdLessonBeforeLunchTurnout', 'PELessonBeforeLunchTurnout', 'Cloudiness', 'Temperature', 'WindSpeed']
othercols = ['LocationSID', 'MealCountMA', 'TET', 'PackedLunchCount', 'PELessonBeforeLunchTurnout', 'Cloudiness', 'Temperature']

Xmenu = dftrain[menucols].values
Xother = dftrain[othercols].values
X = np.concatenate((Xmenu, Xother), axis=1)
y = np.array(dftrain['MealCount'])
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)
modelMeal = linear_model.LinearRegression()
modelMeal.fit(X_scaled,y)
dftrain['PredictedMealCount'] = modelMeal.predict(X_scaled)
#%%
dftrainwaste = df.dropna(subset=['WasteTotalKg'])
Xmenu = dftrainwaste[menucols].values
Xother = dftrainwaste[othercols].values
X = np.concatenate((Xmenu, Xother), axis=1)
#X = Xmenu
y = np.array(dftrainwaste['WasteTotalKg'])

modelWaste = linear_model.LinearRegression()
modelWaste.fit(X,y)
dftrainwaste['PredictedWasteTotalKg'] = modelWaste.predict(X)

#%%
dftest = df.loc[(df['Date'] >= today) & (df['Date'] < today + pd.DateOffset(days=8))]
dftest['MealCountMA'].fillna(method='ffill', inplace=True)
Xtestmenu = dftest[menucols].values
Xtestother = dftest[othercols].values
Xtest = np.concatenate((Xtestmenu, Xtestother), axis=1)
Xtest_scaled = scaler.transform(Xtest)
dftest['PredictedMealCount'] = modelMeal.predict(Xtest_scaled)
#Xtest = Xtestmenu
dftest['PredictedWasteTotalKg'] = modelWaste.predict(Xtest_scaled)
#%%
predictmeals = dftest['PredictedMealCount'].values
predictwaste = dftest['PredictedWasteTotalKg'].values

print('Predicted meals: ', predictmeals)
print('Predicted waste: ', predictwaste)
#%%

dfplot = dftrain.loc[dftrain['LocationSID']==1]
dfplotwaste = dftrainwaste.loc[dftrainwaste['LocationSID']==1]
#%%
fig, ax = plt.subplots()
plt.plot(dfplot['Date'], dfplot['PredictedMealCount'].values, color='red', label='PredictedMealCount')
plt.plot(dfplot['Date'], dfplot['MealCount'].values, color='black', label='MealCount')
plt.legend(fontsize=14)
AImodelError = mean_absolute_error(dfplot['MealCount'].values, dfplot['PredictedMealCount'].values)
NaiveError = mean_absolute_error(dfplot['MealCount'].values,
                          np.repeat(np.mean(dfplot['MealCount'].values),
                                        len(dfplot['MealCount'].values)))
plt.title('Meal Count \n' + 'AI model mean error %.2f' % AImodelError + '\n Naive model error %.2f' % NaiveError, fontsize=14 )

#%%
fig, ax = plt.subplots()
plt.plot(dfplotwaste['Date'], dfplotwaste['PredictedWasteTotalKg'].values, color='red', label='PredictedWasteTotalKg')
plt.plot(dfplotwaste['Date'], dfplotwaste['WasteTotalKg'].values, color='black', label='WasteTotalKg')
plt.legend(fontsize=14)
AImodelError = mean_absolute_error(dfplotwaste['WasteTotalKg'].values, dfplotwaste['PredictedWasteTotalKg'].values)
NaiveError = mean_absolute_error(dfplotwaste['WasteTotalKg'].values,
                          np.repeat(np.mean(dfplotwaste['WasteTotalKg'].values),
                                        len(dfplotwaste['WasteTotalKg'].values)))
plt.title('Waste total kg \n' + 'AI model mean error %.2f' % AImodelError + '\n Naive model error %.2f' % NaiveError,fontsize=14 )

#%%
forest = RandomForestRegressor()
forest.fit(X,y)
importances = forest.feature_importances_

coeffs = modelMeal.coef_
input_vars = []
for i in range(0,len(menucols)):
    input_vars.append(menucols[i][9:])
for i in range(0,len(othercols)):
    input_vars.append(othercols[i])

corr_vars = []
for i in range(0,len(menucols)):
    corr_vars.append(menucols[i])
for i in range(0,len(othercols)):
    corr_vars.append(othercols[i])
corrs = []
corrswaste = []
for var in corr_vars:
    dftemp = df.dropna(subset=[var, 'MealCount'])
    corrs.append(np.corrcoef(dftemp[var].values, dftemp['MealCount'].values)[0][1])
    dftemp = df.dropna(subset=[var, 'WasteTotalKg'])    
    corrswaste.append(np.corrcoef(dftemp[var].values, dftemp['WasteTotalKg'].values)[0][1])

dfcorrs = pd.DataFrame()
dfcorrs['Variable'] = input_vars
dfcorrs['CorrelationToMealCount'] = corrs
dfcorrs['CorrelationToWasteTotalKg'] = corrswaste

fig, ax = plt.subplots(figsize=(10,10))
#plt.barh(dfcorrs['Variable'], dfcorrs['CorrelationToMealCount'], color='orange')
plt.barh(input_vars, coeffs, color='orange')
#plt.barh(input_vars, importances, color='orange')
plt.title('Feature importances for MealCount predictions', fontsize=14)
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=12)
fig.tight_layout()

fig, ax = plt.subplots()
plt.barh(dfcorrs['Variable'], dfcorrs['CorrelationToMealCount'], color='orange')
#plt.barh(input_vars, coeffs, color='orange')
#plt.barh(input_vars, importances, color='orange')
plt.title('Correlations for MealCount', fontsize=14)
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=14)
fig.tight_layout()

print(r2_score(np.array(dftrain['MealCount']), modelMeal.predict(X_scaled)))

#%%
print(len(df.loc[(df['LocationSID']==2) & (df['MealCount']>0)]))
print(len(df.loc[(df['LocationSID']==2) & (df['WasteTotalKg']>0)]))

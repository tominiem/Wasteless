import pandas as pd
import numpy as np

daydict = {'ma':0, 'ti':1, 'ke':2, 'to':3, 'pe':4, 'la':5, 'su':6}
dayinvdict = {0:'ma', 1:'ti', 2:'ke', 3:'to', 4:'pe', 5:'la', 6:'su'}
#firstweek = {2016:31, 2017:29, 2018:31, 2019:32}
months = ['elokuu', 'syyskuu', 'lokakuu', 'marraskuu', 'joulukuu',
          'tammikuu', 'helmikuu', 'maaliskuu', 'huhtikuu', 'toukokuu']

df = pd.DataFrame()
for y in range(2016,2019):
    dfrl = pd.read_excel('./ruokalistat/ruokalista_'+str(y)+'-'+str(y+1)+'.xlsx')
    fw = int(dfrl[1][5].split(',',1)[0])
    fc = ((52-fw) % 5)
    for i in range(0,len(months)):
        if i<5:
            dftemp = pd.read_excel('./suoritteet/YL KOULU  '+months[i]+' '+str(y)+'.xlsx',
                    skiprows=2, usecols=[1,2,3], names=['Day', 'Date', 'Suoritteet'], parse_dates=True)
            dftemp = dftemp.dropna(subset=['Day'])
            dftemp['Viikko'] = dftemp['Date'].dt.week
            dftemp = dftemp[dftemp['Day'] != 'la']
            dftemp = dftemp[dftemp['Day'] != 'su']
            dftemp['Ruoka'] = dftemp.apply(lambda row: dfrl[((row['Viikko']-fw) % 5)+1][daydict[row['Day']]], axis=1)
        else:
            dftemp = pd.read_excel('./suoritteet/YL KOULU  '+months[i]+' '+str(y+1)+'.xlsx',
                    skiprows=2, usecols=[1,2,3], names=['Day', 'Date', 'Suoritteet'], parse_dates=True)
            dftemp = dftemp.dropna(subset=['Day'])
            dftemp['Viikko'] = dftemp['Date'].dt.week
            dftemp = dftemp[dftemp['Day'] != 'la']
            dftemp = dftemp[dftemp['Day'] != 'su']
            dftemp['Ruoka'] = dftemp.apply(lambda row: dfrl[(((row['Viikko'] % 5)+fc) % 5)+1][daydict[row['Day']]], axis=1)
        dftemp['Ruokalaji'] = dftemp.apply(lambda row: row['Ruoka'].split(',', 1)[0].split('/', 1)[0].split(' ', 1)[0], axis=1)
        df = pd.concat([df,dftemp], ignore_index=True)
#%%
dfrl = pd.read_excel('./ruokalistat/ruokalista_2019-2020.xlsx', names=['Viikko', 1,2,3,4,5])
fw = int(dfrl[1][5].split(',',1)[0])
fc = ((52-fw) % 5)
dftemp = pd.read_excel('./suoritteet/YL KOULU  elokuu 2019.xlsx',
                    skiprows=2, usecols=[1,2,3], names=['Day', 'Date', 'Suoritteet'], parse_dates=True)
dftemp = dftemp.dropna(subset=['Day'])
dftemp['Viikko'] = dftemp['Date'].dt.week
dftemp = dftemp[dftemp['Day'] != 'la']
dftemp = dftemp[dftemp['Day'] != 'su']
dftemp['Ruoka'] = dftemp.apply(lambda row: dfrl[((row['Viikko']-fw) % 5)+1][daydict[row['Day']]], axis=1)
dftemp['Ruokalaji'] = dftemp.apply(lambda row: row['Ruoka'].split(',', 1)[0].split('/', 1)[0].split(' ', 1)[0], axis=1)
df = pd.concat([df,dftemp], ignore_index=True)

dftemp = pd.read_excel('./suoritteet/YL KOULU  syyskuu 2019.xlsx',
                    skiprows=2, usecols=[1,2,3], names=['Day', 'Date', 'Suoritteet'], parse_dates=True)
dftemp = dftemp.dropna(subset=['Day'])
dftemp['Viikko'] = dftemp['Date'].dt.week
dftemp = dftemp[dftemp['Day'] != 'la']
dftemp = dftemp[dftemp['Day'] != 'su']
dftemp['Ruoka'] = dftemp.apply(lambda row: dfrl[((row['Viikko']-fw) % 5)+1][daydict[row['Day']]], axis=1)
dftemp['Ruokalaji'] = dftemp.apply(lambda row: row['Ruoka'].split(',', 1)[0].split('/', 1)[0].split(' ', 1)[0], axis=1)
df = pd.concat([df,dftemp], ignore_index=True)
#%%
dftemp = pd.read_excel('./suoritteet/YL KOULU  lokakuu 2019-helmikuu 2020.xlsx',
                    skiprows=3, usecols=[2,3], names=['Date', 'Suoritteet'], parse_dates=True)
dftemp = dftemp[0:152]
dftemp['Day'] = dftemp.apply(lambda row: dayinvdict[row['Date'].weekday()], axis=1)
dftemp = dftemp.dropna(subset=['Day'])
dftemp['Viikko'] = dftemp['Date'].dt.week
dftemp = dftemp[dftemp['Day'] != 'la']
dftemp = dftemp[dftemp['Day'] != 'su']
dftemp['Ruoka'] = dftemp.apply(lambda row: dfrl[((row['Viikko']-fw) % 5)+1][daydict[row['Day']]] if row['Viikko'] > 25 else dfrl[(((row['Viikko'] % 5)+fc) % 5)+1][daydict[row['Day']]], axis=1)
dftemp['Ruokalaji'] = dftemp.apply(lambda row: row['Ruoka'].split(',', 1)[0].split('/', 1)[0].split(' ', 1)[0], axis=1)
df = pd.concat([df,dftemp], ignore_index=True)
#%%
dftemp = pd.DataFrame()
dftemp['Date'] = pd.date_range(start='2020-02-22', end='2020-03-16')
dftemp['Day'] = dftemp.apply(lambda row: dayinvdict[row['Date'].weekday()], axis=1)
dftemp['Viikko'] = dftemp['Date'].dt.week
dftemp = dftemp[dftemp['Day'] != 'la']
dftemp = dftemp[dftemp['Day'] != 'su']
dftemp['Ruoka'] = dftemp.apply(lambda row: dfrl[((row['Viikko']-fw) % 5)+1][daydict[row['Day']]] if row['Viikko'] > 25 else dfrl[(((row['Viikko'] % 5)+fc) % 5)+1][daydict[row['Day']]], axis=1)
dftemp['Ruokalaji'] = dftemp.apply(lambda row: row['Ruoka'].split(',', 1)[0].split('/', 1)[0].split(' ', 1)[0], axis=1)
#%%
df = pd.concat([df,dftemp], ignore_index=True)
#%%
df.replace(to_replace=['\nTalon', 'Hedelmäinen', 'Makkara-', 'Kala(kasvis)keitto', 'Ohra-riisipuuro', 'Suikalelihakastike', 'Kappalekala', 'Broileripatukka'],
           value=['Talon', 'Broilerikastike', 'Makkarakeitto', 'Kalakeitto', 'Riisi-ohrapuuro', 'Lihakastike', 'Kalaleike', 'Broilerinugetti'], inplace=True)
#%%
dfhav = pd.read_excel('./havikit/Hävikkiseuranta 20200317_V3.xlsx', skiprows=2,
                    usecols=[0,12], names=['Tuote', 'Havikki'])
dfhav = dfhav.dropna(subset=['Havikki'])
dfhav['Date'] = dfhav.apply(lambda row: pd.to_datetime(row['Tuote'].split(' ',1)[1], dayfirst=True), axis=1)
dfhav.set_index('Date', drop=True, inplace=True)
df['Havikki'] = df.apply(lambda row: dfhav['Havikki'][row['Date']] if row['Date'] in dfhav.index else 0, axis=1)

#%%

dfretket = pd.read_excel('Retket.xlsx', skiprows=2, usecols=[0,1], names=['Date', 'Number'], parse_dates=True)
dfretket = dfretket.dropna(subset=['Date'])
dfretket.set_index('Date', drop=True, inplace=True)
df['Evaat'] = df.apply(lambda row: dfretket['Number'][row['Date']] if row['Date'] in dfretket.index else 0, axis=1)

#%%

dfsaa = pd.read_csv('KauhavaSaa2016-2020.csv', encoding='latin')
dfsaa['Date'] = pd.to_datetime({'year':dfsaa['Vuosi'], 'month':dfsaa['Kk'], 'day':dfsaa['Pv']})
dfsaa1 = dfsaa.groupby('Date').mean()
df['Sade'] = df.apply(lambda row: dfsaa1['Sateen intensiteetti (mm/h)'][row['Date']], axis=1)
df['Lampotila'] = df.apply(lambda row: dfsaa1['Ilman lämpötila (degC)'][row['Date']], axis=1)
df['Tuuli'] = df.apply(lambda row: dfsaa1['Tuulen nopeus (m/s)'][row['Date']], axis=1)
df['Nakyvyys'] = df.apply(lambda row: dfsaa1['Näkyvyys (m)'][row['Date']], axis=1)

#%%

infmap = {'matala':0, 'matala*':0, 'kohtalainen':1, 'korkea':2, 'hyvin korkea':3}
dfterveys = pd.read_excel('./terveys/Influenssakaudet EP-lla 2015-2019_allekkain.xlsx', skiprows=1)
dfterveys['InfluenssaRiski'] = dfterveys.apply(lambda row: infmap[row['Influenssa-aktiivisuus']], axis=1)

df['InfluenssaRiski'] = df.apply(lambda row: dfterveys[(dfterveys['Vuosi'] == row['Date'].year) & (dfterveys['Viikot'] == row['Viikko'])]['InfluenssaRiski'].iloc[0], axis=1)
#%%

df['ViikkoCont'] = df.apply(lambda row: row['Viikko'] -52, axis=1)

#%%
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

dfs = df.dropna()
dfs = dfs[dfs['Suoritteet'] < 600]
dfs = dfs[dfs['Date'] > pd.to_datetime('2017-11-27')]

Xruokalajit = pd.get_dummies(dfs['Ruokalaji'])
Xmuut = dfs[['Evaat', 'InfluenssaRiski']].values
X = np.concatenate((Xruokalajit, Xmuut), axis=1)
y = np.array(dfs['Suoritteet'])
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.2)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
dfs['EnnusteLR'] = model.predict(X)
#%%

suoritemean = y_train.mean()
print('LR tarkkuus train:', mean_absolute_error(y_train, model.predict(X_train)))
print('LR tarkkuus test:', mean_absolute_error(y_test, model.predict(X_test)))
print('Naivin mallin tarkkuus:', mean_absolute_error(y_test, np.repeat(suoritemean, len(y_test))))

#%%
dfh = df.dropna(subset=['Havikki'])
dfh = dfh[dfh['Suoritteet'] < 600]
dfh = dfh[dfh['Date'] > pd.to_datetime('2017-11-27')]

Xruokalajit = pd.get_dummies(dfh['Ruokalaji'])
Xmuut = dfh[['Evaat','InfluenssaRiski']].values
X = np.concatenate((Xruokalajit, Xmuut), axis=1)
y = np.array(dfh['Havikki'])
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.2)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
dfh['EnnusteLR'] = model.predict(X)

suoritemean = y_train.mean()
print('LR havikki tarkkuus train:', mean_absolute_error(y_train, model.predict(X_train)))
print('LR havikki tarkkuus test:', mean_absolute_error(y_test, model.predict(X_test)))
print('Naivin havikki mallin tarkkuus:', mean_absolute_error(y_test, np.repeat(suoritemean, len(y_test))))


#%%

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

ruuat = list(df['Ruokalaji'].unique())
means = []
for ruoka in ruuat:
    means.append(df[df['Ruokalaji']==ruoka]['Suoritteet'].mean())
plt.figure(figsize=(10,10))
plt.barh(ruuat, means)
plt.title('Suoritteiden keskiarvo kullekin ruokalajille')
#%%
plt.figure()
plt.plot(dfs['Date'].values, dfs['Suoritteet'].values, color='black', label='Toteutuneet suoritteet')
plt.plot(dfs['Date'].values, dfs['EnnusteLR'].values, color='red', label='Ennustetut suoritteet')
plt.legend()
#%%

dfyhd = pd.read_csv('yhdistetty_aineisto.csv', encoding='latin')
X = dfyhd.drop(['Lounas koulut', 'pvm'], axis=1)
y = dfyhd['Lounas koulut'].values
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.2)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
dfyhd['EnnusteLR'] = model.predict(X)

suoritemean = y_train.mean()
print('LR tarkkuus train:', mean_absolute_error(y_train, model.predict(X_train)))
print('LR tarkkuus test:', mean_absolute_error(y_test, model.predict(X_test)))
print('Naivin mallin tarkkuus:', mean_absolute_error(y_test, np.repeat(suoritemean, len(y_test))))

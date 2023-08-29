import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics
from sklearn.tree import export_text
import os
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

# %matplotlib inline
#os.listdir('../input')

df= pd.read_csv("https://media.githubusercontent.com/media/nishp763/Quality_Prediction_ML/master/dataset.csv", decimal=',')
df.iloc[115024:115038]

df['date']=pd.to_datetime(df['date'])

counts = df.groupby('date').count()
counts
counts[counts['% Iron Feed'] < 180]

hours = pd.Series(df['date'].unique())
hours.index=hours
hours.index

date_range = pd.date_range(start=df.iloc[0,0], end='2017-09-09 23:59:40', freq='20S')
date_range = date_range[6:]

hours_list = hours.index.format()
seconds_list = date_range.format()

new_index = []

for idx in seconds_list:
    if (idx[:13]+':00:00') in hours_list:
        new_index.append(idx)
len(new_index)
len(df)

new_index.remove('2017-04-10 00:00:00')

df['index'] = new_index
df['index'] = pd.to_datetime(df['index'])
df.index = df['index']

df = df.drop('index', axis=1)

df.rename(columns={'date':'datetime hours'}, inplace=True)

unique_avg = df.groupby('datetime hours').nunique().mean()
plt.figure(figsize=(12,8))
unique_avg.plot(kind='bar')
plt.xticks(list(range(len(unique_avg))), list(df.columns[:-1]), rotation='vertical')
plt.ylabel('Frequency of measurements in one hour')
unique_avg
df['% Silica Concentrate'].nunique()

silica_unique = df.groupby('datetime hours').nunique()['% Silica Concentrate']
print('number of hours with more than one measurement for silica concentration:', silica_unique[silica_unique > 1].count())
print('number of hours with one measurement for silica concentration:', silica_unique[silica_unique == 1].count())

df['% Silica Concentrate'].nunique()

silica_unique = df.groupby('datetime hours').nunique()['% Silica Concentrate']
print('number of hours with more than one measurement for silica concentration:', silica_unique[silica_unique > 1].count())
print('number of hours with one measurement for silica concentration:', silica_unique[silica_unique == 1].count())
#we know that values with more than one measurement is interpolated when we plot it and see a perfect line.
plt.plot(df['% Silica Concentrate'][df['datetime hours'] == silica_unique[silica_unique > 1].index[0]])
interpolated_hours = silica_unique[silica_unique > 1].index.format()
clean_df = df[~df['datetime hours'].isin(interpolated_hours)]
plt.plot(clean_df.index, clean_df['% Iron Feed'])
plt.plot(clean_df.index, clean_df['% Silica Feed'])

print(len(clean_df.groupby('datetime hours').mean()))
print(clean_df['% Iron Feed'].nunique())
print(clean_df['% Silica Feed'].nunique())
print(clean_df['% Silica Concentrate'].nunique())

def get_unique(column):

    df= pd.DataFrame()

    uv_list = list(column.unique())
    count_list = []

    for i in uv_list:
        count_list.append(len(column[column==i]))

    df['unique_values'] = uv_list
    df['count'] = count_list

    return df

if_unique = get_unique(clean_df['% Iron Feed']).sort_values('count', ascending=False)
sf_unique = get_unique(clean_df['% Silica Feed']).sort_values('count', ascending=False)
for i in range(6):
    clean_df['% Silica Feed'][clean_df['% Silica Feed']==sf_unique.iloc[i,0]].plot()
    clean_df['% Iron Feed'][clean_df['% Iron Feed']==if_unique.iloc[i,0]].plot()
    plt.show()
clean_df.groupby([clean_df.index.date, clean_df.index.hour]).mean()
dirty_idx = []

for i in range(4):
    dirty_idx.extend(clean_df['% Silica Feed'][clean_df['% Silica Feed'] == sf_unique.iloc[i,0]].index.format())
clean_df = clean_df[~clean_df.index.isin(dirty_idx)]

clean_df['% Silica Feed'].plot()
clean_df['% Iron Feed'].plot()

len(clean_df)/(180*24)
clean_df['% Silica Concentrate'].plot()
clean_df['time'] = list(range(0, len(clean_df)))
sns.lineplot(data=clean_df, x='time', y='% Iron Feed')
sns.lineplot(data=clean_df, x='index', y='% Iron Feed')
clean_df['Iron Feed lag'] = clean_df['% Iron Feed'].shift(1)
sns.scatterplot(data=clean_df, x='Iron Feed lag', y='% Iron Feed')
clean_df[clean_df['Iron Feed lag']!=clean_df['% Iron Feed']][['Iron Feed lag', '% Iron Feed', '% Silica Concentrate']]
sns.histplot(data=clean_df['% Iron Feed'], bins=46)
sns.histplot(data=clean_df['% Silica Concentrate'], bins=46)
sns.scatterplot(data=clean_df, x='% Silica Concentrate', y='% Silica Feed')
(clean_df['% Iron Feed'].max() - clean_df['% Iron Feed'].min())/(0.5)
clean_df['% Iron Feed'][clean_df['% Iron Feed']>62].count()
clean_df['% Iron Feed'].count()
clean_df.groupby([clean_df.index.month, clean_df.index.day, clean_df.index.hour])['% Silica Concentrate'].std().plot()
clean_df.groupby('datetime hours')['% Silica Concentrate'].nunique().mean()
df1 = clean_df.groupby([clean_df.index.month, clean_df.index.day, clean_df.index.hour]).mean()
df1.corr()['% Silica Concentrate'].sort_values()

df1 = df1.reset_index(drop=True)
df1.drop(['time', 'Iron Feed lag'], axis=1, inplace=True)


plt.figure(figsize=(20, 20))
p = sns.heatmap(df1.corr(), annot=True);

left_data = df1.groupby(df1.index)['% Silica Concentrate'].mean()
right_data = df1.groupby(df1.index)['% Silica Feed'].mean()
fig, ax_left = plt.subplots(figsize=(15,5))
ax_right = ax_left.twinx()
ax_left.plot(left_data, label = "Silicie Final", color='black')
ax_right.plot(right_data, label = "Silicie Inicial", color='red')
fig.legend()

X = df1.drop(['Flotation Column 05 Air Flow', 'Flotation Column 07 Air Flow',
       'Flotation Column 05 Level','% Silica Concentrate'], axis=1)
y = df1['% Silica Concentrate']

#X_scaled_df.shape
print(X.shape)
X.head()

X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y, test_size = 0.15, random_state = 0)

print(X_test1.shape)
print(y_train1.shape)

scaler = StandardScaler()
scaler.fit(X_train1)
X_train1 = scaler.transform(X_train1)
X_test1 = scaler.transform(X_test1)

y_train1.shape

svr_rbf = SVR(kernel = 'rbf', gamma = 'auto', C=3, epsilon=0.15, verbose=0)
svr_rbf=svr_rbf.fit(X_train1,y_train1)

print(svr_rbf.score(X_test1,y_test1))
print(np.sqrt(metrics.mean_squared_error(y_test1,svr_rbf.predict(X_test1))))
y_pred = svr_rbf.predict(X_test1)

print('%RMSE=',((metrics.mean_squared_error(y_test1, y_pred)**(1/2) / df1['% Silica Concentrate'].mean())*100), '%')

mape = metrics.mean_absolute_percentage_error(y_test1, y_pred)
print("MAPE=", mape)
r2 = metrics.r2_score(y_test1,y_pred)
print('R2 = ',r2)

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (20, 10))
ax = fig.add_subplot(111)
ax.set(title = "Modelo de Support Vector Regression", xlabel = "Valor Y_train", ylabel = "Valor Y_test")
ax.scatter(y_test1, y_pred)
ax.plot([0, max(y_test1)], [0, max(y_pred)], color = 'r')
fig.show();


plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Prediccion')
plt.scatter(range(len(y_test1)), y_test1, color='orange', label='Real')
plt.legend(loc='best', frameon=True)
plt.show()

def evaluate(predictions, test_data):
    errors = abs(predictions - test_data)
    mape = 100 * np.mean(errors / test_data)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f}'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%'.format(accuracy))

    return accuracy
base_accuracy = evaluate(y_pred, y_test1)

svr_rbf.get_params()

import warnings; warnings.simplefilter('ignore')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
rf = SVR(kernel='rbf')
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

scaler = MinMaxScaler()
X_tr_scale=scaler.fit_transform(X_train1)

param_grid = {'cache_size': [190,210], 'epsilon': [0.15,2],'C':[0,3]}


model = SVR(kernel='rbf')

#using smaller number of splits for speed
kfold = KFold(n_splits = 20)
grid_search = GridSearchCV(model, param_grid, cv = kfold, scoring = 'neg_mean_squared_error')
grid_search.fit(X_tr_scale, y_train1)

from sklearn.metrics import mean_squared_error

Final_Model = grid_search.best_estimator_
Final_Model.fit(X_tr_scale, y_train1)

X_te_scale = scaler.transform(X_test1)
y_pred_tuned = Final_Model.predict(X_te_scale)
Final_mse = mean_squared_error(y_test1, y_pred)
Final_rmse = np.sqrt(Final_mse)
print('RMSE: {0:.3f}'.format(Final_rmse))
print('R2: {0:.3f}'.format(Final_Model.score(X_te_scale, y_test1)))

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(svr_rbf, file)

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

import pandas as pd

# Reading data
features = pd.read_pickle(r'features.pkl')
features.reset_index(inplace=True)
features.rename(columns={'index': 'datetime', 0 :'vendor_0',1 :'vendor_1', 2 :'vendor_2' }, inplace=True)
target = pd.read_pickle(r'target.pkl')
target.reset_index(inplace=True)
target.rename(columns={'index': 'datetime', 0 : 'prod'}, inplace=True)

# calculating bias
df = target
df = df.merge(features, on='datetime')
df[['vendor_0','vendor_1','vendor_2']] =  df[['vendor_0','vendor_1','vendor_2']].astype(float)
# filtering negative/very low production out
df = df[df['prod']>0.001]

# vendor 0
df['bias_0'] = df['vendor_0'] - df['prod']
bias_0 = df['bias_0'].mean()
#MAPE vendor 0
df['bias_0_pct'] = abs(df['bias_0'] / df['prod'])
mape_0 = df['bias_0_pct'].sum() / df['prod'].shape

# vendor 1
df['bias_1'] = df['vendor_1'] - df['prod']
bias_1 = df['bias_1'].mean()
#MAPE vendor 1
df['bias_1_pct'] = abs(df['bias_1'] / df['prod'])
mape_1 = df['bias_1_pct'].sum() / df['prod'].shape

# vendor 2
df['bias_2'] = df['vendor_2'] - df['prod']
bias_2 = df['bias_2'].mean()
#MAPE vendor 2
df['bias_2_pct'] = abs(df['bias_2'] / df['prod'])
mape_2 = df['bias_2_pct'].sum() / df['prod'].shape

# testing improved forecast method vendor 0 and 1
df_2 = df[['datetime','prod','vendor_0','vendor_1']]
new = df_2[['vendor_0','vendor_1']].mean(axis = 1) # to prevent the slicing error
df_2['vendor_01'] = new
df_2['bias_01'] = df_2['vendor_01'] - df_2['prod']
bias_01 = df_2['bias_01'].mean()
#MAPE vendor 01
df_2['bias_01_pct'] = abs(df_2['bias_01'] / df_2['prod'])
mape_01 = df_2['bias_01_pct'].sum() / df_2['prod'].shape


# testing improved forecast vendor 0 and 2
df_3 = df[['datetime','prod','vendor_0','vendor_2']]
new = df_3[['vendor_0','vendor_2']].mean(axis = 1) # to prevent the slicing error
df_3['vendor_02'] = new
df_3['bias_02'] = df_3['vendor_02'] - df_3['prod']
bias_02 = df_3['bias_02'].mean()
#MAPE vendor 02
df_3['bias_02_pct'] = abs(df_3['bias_02'] / df_3['prod'])
mape_02 = df_3['bias_02_pct'].sum() / df_3['prod'].shape

# testing improved forecast vendor 1 and 2
df_4 = df[['datetime','prod','vendor_1','vendor_2']]
new = df_4[['vendor_1','vendor_2']].mean(axis = 1) # to prevent the slicing error
df_4['vendor_12'] = new
df_4['bias_12'] = df_4['vendor_12'] - df_4['prod']
bias_12 = df_4['bias_12'].mean()
#MAPE vendor 12
df_4['bias_12_pct'] = abs(df_4['bias_12'] / df_4['prod'])
mape_12 = df_4['bias_12_pct'].sum() / df_4['prod'].shape

# testing improved forecast vendor 0, 1 and 2
df_5 = df[['datetime','prod','vendor_0','vendor_1','vendor_2']]
new = df_5[['vendor_0','vendor_1','vendor_2']].mean(axis = 1) # to prevent the slicing error
df_5['vendor_012'] = new
df_5['bias_012'] = df_5['vendor_012'] - df_5['prod']
bias_012 = df_5['bias_012'].mean()
#MAPE vendor 012
df_5['bias_012_pct'] = abs(df_5['bias_012'] / df_5['prod'])
mape_012 = df_5['bias_012_pct'].sum() / df_5['prod'].shape

# weekly production
weekly = target.groupby(target.datetime.dt.week)['prod'].sum().reset_index()
weekly['average_prod'] = weekly['prod'] / 168
average_production = weekly['prod'].mean()
weekly['cap'] = target['prod'].max()*2
weekly['cap_factor'] = weekly['average_prod'] / weekly['cap']
cap_factor_avg = weekly['cap_factor'].mean()

print("stop")

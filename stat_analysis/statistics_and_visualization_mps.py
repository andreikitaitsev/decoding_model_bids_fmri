import seaborn as sea
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import statsmodels.formula.api as stm
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

# read correlation data and convey it into pandas dataframe
# 1st line sm, 2nd line stm
model_types=['SM','SM','SM','SM','STM','STM','STM'] 
models = ['pca100_ridge','pca300_ridge','pca300_cca30','ica300_cca30',\
'pca100_ridge','pca300_ridge','pca300_cca30']
subjects = ['sub-01','sub-02','sub-03']#,'sub-04']
data=[]

# order of looping matters!
for model in range(len(models)):
    for subject in range(len(subjects)):
        dat=joblib.load(os.path.join('/data/akitaitsev/decoding_model_bids/decoding_data/', model_types[model],\
            models[model], subjects[subject], ('correlations_'+str(subjects[subject])+'.pkl')))
        
        # check if there are 0 correlations
        inds = np.where(np.isclose(dat, np.zeros_like(dat))) 
        if not len(inds[0])==0:
            print('model ',str(model),'\n subject ',str(subject))
        data.append(dat)
assert [data[0].shape == el.shape for el in data]
pnts4subj = data[0].shape[0]
data=np.concatenate(data, axis=0)


# create long format data table
models = ['pca100_ridge','pca300_ridge','pca300_cca30','ica300_cca30',\
'Pca100_ridge','Pca300_ridge','Pca300_cca30']
subjects_ = [np.tile(el,pnts4subj) for el in subjects]
subjects_= np.concatenate(subjects_)
subjects_=subjects_.tolist()*len(models)
models_=[np.tile(el, pnts4subj*len(subjects)) for el in models]
models_=np.concatenate(models_)
model_types_=[np.tile(el,pnts4subj*len(subjects)) for el in model_types]
model_types_=np.concatenate(model_types_)
data= {'model':models_, 'model_type':model_types_, 'subject':subjects_,'data':data}
df = pd.DataFrame.from_dict(data)
# save dataframe
df.to_csv('/data/akitaitsev/decoding_model_bids/decoding_data/statistics/df_long_cor_mps.csv',\
    index=False)

# plot violin plots
fig, ax = plt.subplots(figsize=(16,9))
sea.violinplot(ax=ax, x='model',y='data', hue='model_type', kind='violin',inner='quartile',hue_order=['SM','STM'],data=df)
plt.show()
fig.savefig('/data/akitaitsev/decoding_model_bids/decoding_data/statistics/violinplot_mps.png', dpi=300)

### Statistical analyis

# GLM
#model_mix=sm.mixedlm('data~C(subject)+C(model_type)+ C(model)', data=df, groups="subject").fit()
#print(model_mix.summary())
#model=sm.GLM(df.data, df[["model","model_type","subject"]]).fit()
#print(model_res.summary())

# 2 way anova repeated measures
aov=pg.mixed_anova(dv='data', between='model_type',within='model',subject='subject', data=df)
aov.to_csv('/data/akitaitsev/decoding_model_bids/decoding_data/statistics/anova_rep_measures_mps.csv')
pg.print_table(aov)

# 3 way anova
model=ols('data~C(model)+C(model_type)+C(subject)', data=df).fit()
anova=sm.stats.anova_lm(model, typ=2)
anova.to_csv('/data/akitaitsev/decoding_model_bids/decoding_data/statistics/anova_3way_mps.csv')
print(anova)


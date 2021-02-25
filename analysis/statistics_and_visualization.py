import seaborn as sea
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols


# read correlation data and convey it into pandas dataframe
model_types=['SM','SM','SM','SM','STM','STM','STM']
models = ['ica300_cca30', 'ica300_ridge', 'pca300_cca30', 'pca300_ridge',\
    'ica100_ridge','pca300_cca30','pca300_ridge'] # 1st line-SM, 2nd-STM
subjects = ['sub-01','sub-02','sub-03']#,'sub-04']
data=[]

# order of looping matters!
for model in range(len(models)):
    for subject in range(len(subjects)):
        dat=joblib.load(os.path.join('/data/akitaitsev/data1/decoding_data5/', model_types[model],\
            models[model], subjects[subject], ('correlations_'+str(subjects[subject])+'.pkl')))
        
        # check if there are 0 correlations
        inds = np.where(np.isclose(dat, np.zeros_like(dat))) 
        if not len(inds[0])==0:
            print('model ',str(model),'\n subject ',str(subject))
        data.append(dat)
assert [data[0].shape == el.shape for el in data]
pnts4subj = data[0].shape[0]
data=np.concatenate(data, axis=0)

# change model name otherwise seaborn groups torether similar mmodel names from SM and STM
models = ['ica300_cca30', 'ica300_ridge', 'pca300_cca30', 'pca300_ridge',\
    'Ica100_ridge','Pca300_cca30','Pca300_ridge'] # STM, add some more models later

# create long format data table
subjects_ = [np.tile(el,pnts4subj) for el in subjects]
subjects_= np.concatenate(subjects_)
subjects_=subjects_.tolist()*len(models)
models_=[np.tile(el, pnts4subj*len(subjects)) for el in models]
models_=np.concatenate(models_)
model_types_=[np.tile(el,pnts4subj*len(subjects)) for el in model_types]
model_types_=np.concatenate(model_types_)
data= {'model':models_, 'model_type':model_types_, 'subject':subjects_,'data':data}
df = pd.DataFrame.from_dict(data)

# plot violin plots
fig, ax = plt.subplots(figsize=(16,9))
sea.violinplot(ax=ax, x='model',y='data', hue='model_type', kind='violin', data=df)
fig.savefig('/data/akitaitsev/data1/decoding_data5/violinplot.png', dpi=300)

### Statistical analyis
# 2 ways anova on data averaged across runs

model=ols('data~C(model)+C(model_type)+C(subject)', data=df).fit()
summary_table= model.summary().as_csv()
summary_anova=sm.stats.anova_lm(model, typ=2).summary().as_csv()





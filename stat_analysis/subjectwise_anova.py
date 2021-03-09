import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

out_path = '/data/akitaitsev/decoding_model_bids/decoding_data/statistics/'
## MPS
MPS = pd.read_csv('/data/akitaitsev/decoding_model_bids/decoding_data/statistics/df_long_cor_mps.csv')
mps_s1 = MPS.loc[MPS["subject"]=="sub-01"]
mps_s2 = MPS.loc[MPS["subject"]=="sub-02"]
mps_s3 = MPS.loc[MPS["subject"]=="sub-03"]

mps_s1.pop("subject")
mps_s2.pop("subject")
mps_s3.pop("subject")

# z score for ANOVA
mps_s1["data"] = stats.zscore(mps_s1["data"])
mps_s2["data"] = stats.zscore(mps_s2["data"])
mps_s3["data"] = stats.zscore(mps_s3["data"])

mps_model1=ols('data~C(model)+C(model_type)', data=mps_s1).fit()
mps_model2=ols('data~C(model)+C(model_type)', data=mps_s2).fit()
mps_model3=ols('data~C(model)+C(model_type)', data=mps_s3).fit()

mps_anova1=sm.stats.anova_lm(mps_model1, typ=2)
mps_anova2=sm.stats.anova_lm(mps_model2, typ=2)
mps_anova3=sm.stats.anova_lm(mps_model3, typ=2)

## spectr
spectr = pd.read_csv('/data/akitaitsev/decoding_model_bids/decoding_data/statistics/df_long_cor_spectr.csv')

spectr_s1 = spectr.loc[spectr["subject"]=="sub-01"]
spectr_s2 = spectr.loc[spectr["subject"]=="sub-02"]
spectr_s3 = spectr.loc[spectr["subject"]=="sub-03"]

spectr_s1.pop("subject")
spectr_s2.pop("subject")
spectr_s3.pop("subject")

# z score for ANOVA
spectr_s1["data"] = stats.zscore(spectr_s1["data"])
spectr_s2["data"] = stats.zscore(spectr_s2["data"])
spectr_s3["data"] = stats.zscore(spectr_s3["data"])

spectr_model1=ols('data~C(model)+C(model_type)', data=spectr_s1).fit()
spectr_model2=ols('data~C(model)+C(model_type)', data=spectr_s2).fit()
spectr_model3=ols('data~C(model)+C(model_type)', data=spectr_s3).fit()

spectr_anova1=sm.stats.anova_lm(spectr_model1, typ=2)
spectr_anova2=sm.stats.anova_lm(spectr_model2, typ=2)
spectr_anova3=sm.stats.anova_lm(spectr_model3, typ=2)

# save data
mps_anova1.to_csv(os.path.join(out_path, 'mps_2way_anova_s1.csv'))
mps_anova2.to_csv(os.path.join(out_path, 'mps_2way_anova_s2.csv'))
mps_anova3.to_csv(os.path.join(out_path, 'mps_2way_anova_s3.csv'))

spectr_anova1.to_csv(os.path.join(out_path, 'spectr_2way_anova_s1.csv'))
spectr_anova2.to_csv(os.path.join(out_path, 'spectr_2way_anova_s2.csv'))
spectr_anova3.to_csv(os.path.join(out_path, 'spectr_2way_anova_s3.csv'))


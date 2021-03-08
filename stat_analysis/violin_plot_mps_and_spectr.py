import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

table_mps = pd.read_csv('/data/akitaitsev/decoding_model_bids/decoding_data/statistics/df_long_cor_mps.csv')
mps_col = ['mps']*table_mps.shape[0]
table_mps["feature"] = mps_col

table_spectr=pd.read_csv('/data/akitaitsev/decoding_model_bids/decoding_data/statistics/df_long_cor_spectr.csv')
spectr_col=['spectr']*table_spectr.shape[0]
table_spectr["feature"] = spectr_col

table_general=pd.concat([table_mps, table_spectr], ignore_index=True)
#sea.set(rc={'figure.figsize':(16,9)})
#sea.set_style("whitegrid")
fig=sea.catplot(x='data', y='model', hue='model_type', kind='violin', inner='quartile',\
    hue_order= ['SM','STM'], split=True, col='feature',  data=table_general)
plt.show()
fig.savefig('/data/akitaitsev/decoding_model_bids/decoding_data/statistics/violinplot_general1.png', dpi=300)

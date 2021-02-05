
#! /usr/bin/env python3
'''Script to run the analysis using decoding_model_v6_script_based.py'''
import os
from sklearn.pipeline import Pipeline
import decoding_v6_script_based as dec
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer


## define parameters for decoding model
inp_dir = '/data/akitaitsev/data1/lagged/'
stim_param_dir = '/data/akitaitsev/data1/raw_data/processed_stimuli/'
model_config = {'subjects': ['01']}
out_dir = '/data/akitaitsev/data1/decoding_data3_script_based/STM_test/'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
### spatial temporal models
print('Running spatial temporal models')

sp_dec = PCA(n_components=300)
myRidge_config = {"alphas":[0,5,5]}
lag_par=4
lag = lambda x : dec.lag(x, lag_par)
tmp_dec = Pipeline([('lagging', FunctionTransformer(lag)), ('decoding', dec.myRidge(**myRidge_config)) ])
#tmp_dec = dec.temporal_decoder(dec.myRidge, myRidge_config, lag_par)
sp_tmp_model = Pipeline([('sp',sp_dec),('tmp', tmp_dec)]) 

dec.run_decoding(inp_dir, out_dir, stim_param_dir, model_config, sp_tmp_model)


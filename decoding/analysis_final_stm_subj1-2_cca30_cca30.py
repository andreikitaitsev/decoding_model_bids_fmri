#! /usr/bin/env python3
'''Script to run the analysis using decoding_model_v7.py'''
import os
from sklearn.pipeline import Pipeline
import decoding_v7 as dec
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA

### define parameters for decoding model
inp_dir = '/data/akitaitsev/data1/lagged/'
stim_param_dir = '/data/akitaitsev/data1/raw_data/processed_stimuli/'
model_configs = [{'subjects': ['01']}, {'subjects':['02']}]#, {'subjects':['03']},{'subjects':['04']}]
out_dir = '/data/akitaitsev/data1/decoding_data5/'


### spatial temporal models
print('Running spatial temporal models')
# create spatial-temporal decoders
lag_par = 3 # determined by GridSearch
spatial_decoders = [CCA]
spatial_decoders_configs = [ {'n_components':30}]
spatial_decoders = dec.invoke_decoders(spatial_decoders, spatial_decoders_configs)

temporal_decoders = [dec.myRidge, CCA]
temporal_decoders_configs = [ {'n_components':30}]
folder_names = ['cca30_cca30']

for subject in range(len(model_configs)):
    folder_cntr = 0 # folder counter
    for sp_dec in range(len(spatial_decoders)):
        for tmp_dec in range(len(temporal_decoders)):
            print('Running STM '+folder_names[folder_cntr])
            temporal_decoder = dec.temporal_decoder(temporal_decoders[tmp_dec], temporal_decoders_configs[tmp_dec], lag_par)
            spatial_decoder = spatial_decoders[sp_dec]
            STM = Pipeline([('spatial', spatial_decoder), ('temporal', temporal_decoder)])
            # create output dirs
            out_dir_iter = os.path.join(out_dir, 'STM', folder_names[folder_cntr])
            if not os.path.isdir(out_dir_iter):
                os.makedirs(out_dir_iter)
            # run decoding model
            dec.run_decoding(inp_dir, out_dir_iter, stim_param_dir, model_configs[subject], decoder=STM)
        folder_cntr += 1

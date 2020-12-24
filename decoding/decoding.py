#! /usr/bin/env pyhon3
# Script to run pilot decoding model

import numpy as np 
import sys
import joblib
import os
sys.path.append('/data/akitaitsev/data1/code/decoding/')
from encoding import product_moment_corr, ridge_gridsearch_per_target
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import sklearn.metrics 
import json
import matplotlib.pyplot as plt

__all__ = ['get_ridge_and_staff','reduce_dimensionality','compute_correlation', 'plot_log_mse','reshape_mps',\
        'plot_mps_and_reconstructed_mps', 'decode_single_run','decoding_model']

def get_ridges_and_staff(X, y, scorers, alphas=[1000],  n_splits=8, voxel_selection=True, **kwargs):
    '''Returns ridge regressions trained in a cross-validation on n_splits of the data, scores on the left-out foldsand predicted data.

    Parameters
    ----------
    X : ndarray of shape (samples, features)
    y : ndarray of shape (samples, targets)
    scorers - list of scoring functions to use in (product_moment_corr or any sklearn scoring function 
    alphas : list of floats, optional
             Regularization parameters to be used for Ridge regression
    n_splits : int, optional

    voxel_selection : bool, optional, default True
                      Whether to only use voxels with variance larger than zero.
                      This will set scores for these voxels to zero.
    kwargs : additional arguments transferred to ridge_gridsearch_per_target

    Returns
    ridges -      list of ridges trained on n_splits cross-validation of your data
    scores_dict - dictionary of scores yielded by user-defined scorers
    pred_data -   np 2d array of reconstructed data
    test_inds -   list of test indices 
    -------
    '''
    
    kfold = KFold(n_splits=n_splits)
    ridges = [] 
    scores_dict = {}
    test_inds=[]
    pred_data = [] 
    from collections import defaultdict 
    scores_dict = defaultdict(list) 
    
    if voxel_selection:
        voxel_var = np.var(y, axis=0)
        y = y[:, voxel_var > 0.]
    for train, test in kfold.split(X, y):
        ridges.append(ridge_gridsearch_per_target(X[train], y[train], alphas, **kwargs))
        # loop through scorers
        for ind in range(len(scorers)):
            if scorers[ind] != 'product_moment_corr':
                scorer=eval('sklearn.metrics.'+scorers[ind])
        #TODO allow to feed in the function object!!!
            else:
                scorer = eval(scorers[ind])
        
            if voxel_selection:
                score = np.zeros_like(voxel_var)
                # just temporary: feed multoutput only to sklearn scorers, not to procut_moment_corr 
                if scorers[ind] != 'product_moment_corr':
                    score[voxel_var > 0.] =  scorer(y[test], ridges[-1].predict(X[test]), multioutput='raw_values')
                else:
                    score[voxel_var > 0.] =  scorer(y[test], ridges[-1].predict(X[test]))
            else: 
                if scorers[ind] != 'product_moment_corr':
                    score = scorer(y[test], ridges[-1].predict(X[test]),multioutput='raw_values')
                else:
                    score = scorer(y[test], ridges[-1].predict(X[test]))
            # initialize and update dictionary keys 
            scores_dict[str(scorers[ind])].append(score[:,None])
        # get predicted data
        pred_data.append(ridges[-1].predict(X[test]))
        # keep track of test indices
        test_inds.append(test)
    # concatenate pred_data and scores_dict
    scores_dict = {k:np.concatenate(v, axis=1).T for (k,v) in scores_dict.items()} 
    pred_data = np.concatenate(pred_data, axis=0)
    return ridges, scores_dict, pred_data, test_inds 

def reduce_dimensionality(X, var_explained, examine_variance=False, **kwargs):
    ''' Function applys pca with user specified variance explained to stimulus representation or bold data
    Inputs:
        X - 2d numpy array (samples, features) /(samples/voxels)
        var_explained - float, variabce explained
        examine_variance - logical, whether to run function in interactive mode to manually determine number 
        of components to leave by variance explained plot. Default =Fasle
        kwargs - kwargs for sklearn pca obejct 
    Outputs:
        X_backproj - reduced rand data 
        comp2leave - number of components left
    '''   
    
    from sklearn.decomposition import PCA
    pca=PCA(**kwargs)
    pca.fit(X)
    vars_exp = pca.explained_variance_ratio_
    vars_exp = np.cumsum(vars_exp)
    #[vars_exp_[i]+vars_exp_[i-1] for i in range(vars_exp_.shape[0]) if i !=0]
    comp2leave= None 
    if examine_variance:
        fig, ax = plt.subplots()
        ax = fig.add_subplot(111)
        ax.plot(vars_exp*100, '--ob')
        fig.suptitle('Variance explained')
        ax.set_xlabel('Component number')
        ax.set_ytitle('Variance explained, %')
        
        comp2leave=input('Enter number of components to leave: ')
        pca = PCA(n_components=comp2leave)
        pca.fit(X)
    else:
        comp2leave = vars_exp[vars_exp<var_explained].shape[0]
    pca=PCA(n_components=comp2leave)
    pca.fit(X)
    X_backproj = pca.transform(X)
    return X_backproj, comp2leave

def compute_correlation(real_mps, reconstructed_mps, **kwargs):
    ''' Fucntion computes correlation between real (extracted MPS and MPS reconstruced indecoding model for every time sample.
    Inputs:
        real_mps -          numpy 2d array(times, features) 
        reconstructed_mps - numpy 2d array(times, features) 
    Outputs:
        correlations - 1d numpy array of correlation values (1 per mps timepoins from test_inds).'''
    correlations =[]
    for time in range(reconstructed_mps.shape[0]):
        correlations.append(np.corrcoef(real_mps[time,:], reconstructed_mps[time,:])[0,-1])
    return np.reshape(np.array(correlations), (-1,1))

def plot_mse(mse, mps_time, mps_freqs):
    ''' Function plots mse as imshow and returns figure handle
    Inputs:
        mse - 2d numpy array of reshaped mse
        mps_time - list of strings of mps_time from stimulus parameters json file (output of wav_files....py)
        mps_freqs - list of strings of mps_freqs from stimulus parameters json file (output of wav_files....py)
    Outputs:
        fig - figure handle '''
    fig, ax = plt.subplots()
    fig.suptitle('Mean squared error over all MPSs')
    im = ax.imshow(mse, origin='lower', aspect='auto')
    # transform mps_time and mps_freqs from list of strings to np 1d array to allow indexing
    mps_time = np.array([float(el) for el in mps_time])
    mps_freqs = np.array([float(el) for el in mps_freqs])
    # x labels
    x_inds = np.around(np.linspace(0, len(mps_time), 10, endpoint=False)).astype(int)
    ax.set_xticks(x_inds)
    ax.set_xticklabels(mps_time[x_inds])
    ax.set_xlabel('modulation/s')
    # y labels
    y_inds = np.around(np.linspace(0, len(mps_freqs), 10, endpoint=False)).astype(int)
    ax.set_yticks(y_inds)
    ax.set_yticklabels(mps_freqs[y_inds])
    ax.set_ylabel('modulation/Hz')
    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('MSE')
    return fig

def reshape_mps(mps, mps_shape):
    ''' Fucntion reshapes (flattened) MPS array into its original form (mod/s, mod/Hz, n_mps) in accordance with
    mps_shape from stimulus json parameters file
    Inputs:
        mps - 2d numpy array of shape (n_mps, n_features) - output of make_X_Y function for a single wav file
        mps_shape - tuple(mod/s, mod/Hz) - shape of a single MPS stored in stimulus parameters json file
    Outputs:
        reshaped_mps - 3d numpy array of shape (mod/s, mod/Hz, n_mps) - same shape as original MPS
    '''
    #resahpe to shape (n_mps, mod/s, mod/Hz)
    reshaped_mps = np.reshape(mps, (-1, mps_shape[0], mps_shape[1]))
    # transpose to shape (mod/s, mod/Hz, n_mps)
    reshaped_mps = np.transpose(reshaped_mps, (1,2,0))
    return reshaped_mps

def plot_mps_and_reconstructed_mps(original_mps, reconstructed_mps, mps_time, mps_freqs, fig_title=None, **kwagrs):
    ''' Fucntion plots real and reconstruced MPS on different subplots of one figure
    using matplotlib imshow function
    Inputs:
        original_mps      - numpy 2d array, reshaped to original form original mps
        reconstructed_mps - numpy 2d array, reshaped to original form reconstructed mps
        mps_time          - list or numpy array of mps_time labels (output of feature extractor)
        mps_freqs         - list or numpy array of mps_freqs labels (output of feature extractor)
        fig_title         - str, optional, title of the figure (Default=None)
        kwargs            - arguments for imshow function
    Outputs:
        fig - figure handle'''
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,9)) 
    im1 = ax1.imshow(original_mps, origin='lower', aspect='auto')
    im2 = ax2.imshow(reconstructed_mps, origin='lower', aspect='auto') 
    # transform mps_time and mps_freqs from list of strings to np 1d array to allow indexing
    mps_time = np.array([float(el) for el in mps_time])
    mps_freqs = np.array([float(el) for el in mps_freqs])
    # axes labels
    ax1.title.set_text('Original MPS')
    ax2.title.set_text('Reconstructed MPS')
    # x labels
    x_inds = np.around(np.linspace(0, len(mps_time), 10, endpoint=False)).astype(int)
    ax1.set_xticks(x_inds)
    ax1.set_xticklabels(mps_time[x_inds])
    ax1.set_xlabel('modulation/s')
    ax2.set_xticks(x_inds)
    ax2.set_xticklabels(mps_time[x_inds])
    ax2.set_xlabel('modulation/s')
    # y labels
    y_inds = np.around(np.linspace(0, len(mps_freqs), 10, endpoint=False)).astype(int)
    ax1.set_yticks(y_inds)
    ax1.set_yticklabels(mps_freqs[y_inds])
    ax1.set_ylabel('modulation/s')
    ax2.set_yticks(y_inds)
    ax2.set_yticklabels(mps_freqs[y_inds])
    ax2.set_ylabel('modulation/Hz')
    # colorbar
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('log MPS')
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('log MPS')
    # figure title
    if not fig_title is None:
        fig.suptitle(fig_title)
    return fig
    
def decode_single_run(fmri_path, stimulus_path, alphas, scorers, stim_param_path, do_pca_fmri=False, var_explained=None, **kwargs):
    ''' Runs decoding of single stimulus and fmi run by regressing stimulus representation on bold data via get_ridges_and_staff 
    function (adapted from encoding.py)
    Inputs:
            fmri_path       - str, path to fmri lagged data, output of make_X_Y function
            stimulus_path   - str, path to stimulus representation file
            alphas          - float or list of floats, regularization for ridge, None or list of floats, optional
            scorers         - list of scoring functions to use in get_ridge_plus_scores fucntion
            stim_param_path - str, path to stimulus parameter json file
            do_pca          - logical, whether to do pca on fmri. Default=False
            var_explained   - float, number of components to leave by variance explained for 
                              reduce_dimensionality function. Default = None 
            kwargs          - additional agruments for reduce_dimensionality (for pca)
    Outputs:
            ridges          - analogous to encoding get_ridge_plus_scores
            scores          - dictionary with coefficients of deviation of predicted data from real data (depend on scorer)
            pred_data       - data, reconstructed from ridges
            correlations    - correlations of reconstructed MPS with the original ones (one value per timepoint)
    '''
    # Load data
    with open(fmri_path, 'rb') as fm:
        X = joblib.load(fm) 
    with open(stimulus_path, 'rb') as st:
       y = joblib.load(st)
    if not np.ndim(X) == 2:
        raise ValueError('Inputs shall be 2d numpy array' 
        'fmri shape was'+ str(X.shape)+ 'instead')
    if not np.ndim(y) == 2:
        raise ValueError('Inputs shall be 2d numpy array' 
                'stimulus shape was'+ str(y.shape)+ 'instead')
    with open(stim_param_path, 'r') as par:
        parameters = json.load(par)

    # PCA
    if do_pca_fmri: 
        X,__ = reduce_dimensionality(X, var_explained=var_explained, **kwargs)  
    
    # ridges
    ridges, scores_dict, pred_data, test_inds = get_ridges_and_staff(X, y, scorers, alphas, **kwargs)  
    
    # get mean squared error for every feature and plot it
    mse = np.mean(scores_dict['mean_squared_error'], axis=0)
    resh_mse = lambda x: np.reshape(x, (parameters['mps_shape']))
    mse = resh_mse(mse)
    fig_mse = plot_mse(mse, parameters['mps_time'], parameters['mps_freqs']) 
    
    # compute correlation between reconstructed mps and original mps and plot it
    correlations = compute_correlation(y, pred_data)
    
    fig_cor, ax = plt.subplots(1)
    fig_cor.suptitle('Correlation of original MPS with reconstructed MPS') 
    ax.plot(correlations)
    ax.set_xlabel('MPS times')
    ax.set_ylabel('Correlations per MPS time point')
    
    # reshape original and reconstructed MPS
    orig_mps = reshape_mps(y, parameters['mps_shape'])
    reconstr_mps = reshape_mps(pred_data, parameters['mps_shape'])

    # plot original and reconstructed MPS with best, worst and "medium" correlation values 
    best_mps_ind = np.argmax(np.squeeze(np.abs(correlations)))
    best_mps = np.squeeze(reconstr_mps[:,:,best_mps_ind])
    fig_mps_best = plot_mps_and_reconstructed_mps(orig_mps[:,:,best_mps_ind], best_mps,\
        parameters['mps_time'], parameters['mps_freqs'], 'Best MPS')

    worst_mps_ind = np.argmin(np.squeeze(np.abs(correlations)))
    worst_mps = np.squeeze(reconstr_mps[:,:,worst_mps_ind]) 
    fig_mps_worst = plot_mps_and_reconstructed_mps(orig_mps[:,:,worst_mps_ind], worst_mps,\
        parameters['mps_time'],  parameters['mps_freqs'], 'Worst MPS')

    medium_mps_ind = np.argmin(np.squeeze(np.abs(correlations-np.mean(correlations))))
    medium_mps = np.squeeze(reconstr_mps[:,:,medium_mps_ind])
    fig_mps_medium = plot_mps_and_reconstructed_mps(orig_mps[:,:,medium_mps_ind], medium_mps,\
        parameters['mps_time'], parameters['mps_freqs'], 'Medium MPS')
    figs = [fig_cor, fig_mse, fig_mps_best, fig_mps_worst,fig_mps_medium]  
    return ridges, scores_dict, pred_data, correlations, figs 


def decoding_model(inp_data_dir, out_dir, stim_param_dir, subjects, runs, scorers, alphas, do_pca_fmri=False, var_explained=None, **kwargs):
    ''' Function to run decoding on user-specified runs and subjects.
    Inputs:
           
           inp_data_dir  - str - full path to the directory containing preprocessed subject and fmri data
           out_dir       - str, full path to the directory where you want to save results of decoding model.
                           Function takes care of folder structure and creates folders for each subject if they do not exist
           
           stim_param_dir- str, path to the directory containing stimuli parameters json files for every specified run
           subjects      - list of strings - subject numbers to run decoding on (note that subjects from 0 to 9 shall 
                           have 0 before their number ('01', '09', etc.)!
           runs          - list of lists of integers - numbers of runs to run decoding on (one list per subject)
           alphas        - lisf of floats - regularization parameters for ridge_gridsearch_per_target function
           do_pca_fmri   - boolean, whether to do pca on fmri data. Default = False. 
                           Note, that if do_pca_fmri=True, var_explained argument shall be specified.
           var_explained - number of components to leave by variance explained by them.
           kwargs        - key-value arguments 
    Outputs:
    '''
    #### Input check
    # as model takes long time to run (would a be pity to find a glitch after running model for days:) )
    
    if len(subjects) != len(runs):
        raise ValueError("Number of subjects does not match number of runs you have specified!"
        " Subjects shall be list of subject numbers and runs shall be list of lists of run numbers"
        "for every subject you have specified.")
    
    # loop through subject folders and runs and make sure all the files exist 
    for subj_counter, subj_num in enumerate(subjects):
        subj_folder = 'sub-'+ str(subj_num) 
        subj_folder_path = os.path.join(inp_data_dir, subj_folder)
        if not os.path.isdir(subj_folder_path):
            raise FileNotFoundError
        # Loop through runs of each subject
        for run_num in runs[subj_counter]:
            # get fmri, stimulus and stim_parameters paths and make sure they exist
            fmri = subj_folder + '_task-aomovie_run-' + str(run_num) + '_bold.tsv.gz'
            stim = 'task-aomovie_run-' + str(run_num) + '_stim.tsv.gz'
            stim_param = 'task-aomovie_run-' + str(run_num) + '_stim_parameters.json'
            fmri_path = os.path.join(subj_folder_path, fmri)
            stim_path = os.path.join(subj_folder_path, stim)
            stim_param_path = os.path.join(stim_param_dir, stim_param)
            if not os.path.isfile(stim_param_path):
                raise FileNotFoundError
            if not os.path.isfile(fmri_path):
                raise FileNotFoundError
            if not os.path.isfile(stim_path):
                raise FileNotFoundError
        # Check if subject folders in output dir shall be created
        if not os.path.isdir(os.path.join(out_dir, subj_folder)):
            print('Creating folder '+subj_folder+' in the directory '+out_dir)
            os.mkdir(os.path.join(out_dir, subj_folder))
        elif len(os.listdir(os.path.join(out_dir, subj_folder))) != 0:
            procede = input('Directory ' + os.path.join(out_dir, subj_folder) + ' is not empty!\n Procede? y/n \n')
            if procede == 'n':
                return None
            elif procede =='y':
                pass

    #### Run decoding
    
    for subj_counter, subj_num in enumerate(subjects):
        subj_folder = 'sub-'+ str(subj_num) 
        subj_folder_path = os.path.join(inp_data_dir, subj_folder)
        # Loop through runs of each subject
        for run_num in runs[subj_counter]:
            # get fmri, stimulus and stim_parameters paths and make sure they exist
            fmri = subj_folder + '_task-aomovie_run-' + str(run_num) + '_bold.tsv.gz'
            stim = 'task-aomovie_run-' + str(run_num) + '_stim.tsv.gz'
            stim_param = 'task-aomovie_run-' + str(run_num) + '_stim_parameters.json'
            fmri_path = os.path.join(subj_folder_path, fmri)
            stim_path = os.path.join(subj_folder_path, stim)
            stim_param_path = os.path.join(stim_param_dir, stim_param)
            
            # run decode_single_run
            print('running decode_single_run on subject ',str(subjects[subj_counter]),' run ',str(run_num))
            ridges, scores_dict, pred_data, correlations, figures = decode_single_run(fmri_path,\
                stim_path, alphas, scorers, stim_param_path, do_pca_fmri, var_explained, **kwargs)
            
            # save data into subject-specific folder in the out_dir
            print('saving model data for subject ', str(subjects[subj_counter]), ' run ', str(run_num))
            # name files files
            ridge_name = 'ridges_run-'+str(run_num) + '.pkl'
            scores_dict_name = 'scores_dict_run-' + str(run_num) + '.pkl'
            pred_data_name = 'reconstructed_mps_run-' + str(run_num) + '.pkl'
            correlations_name = 'correlations_run-'+str(run_num) + '.pkl'

            ridges_filename = os.path.join(out_dir, subj_folder, ridge_name) 
            scores_dict_filename =  os.path.join(out_dir, subj_folder, scores_dict_name) 
            pred_data_filename = os.path.join(out_dir, subj_folder, pred_data_name)    
            correlations_filename =  os.path.join(out_dir, subj_folder, correlations_name) 
            # save files
            joblib.dump(ridges, ridges_filename)
            joblib.dump(scores_dict, scores_dict_filename)
            joblib.dump(pred_data, pred_data_filename)
            joblib.dump(correlations, correlations_filename)
            
            # save figures
            # figures = [fig_cor, fig_mse, fig_mps_best, fig_mps_worst, fig_mps_medium] 
            figure_names = ['correlation_run-' + str(run_num)+'.png', 'MSE_run-' + str(run_num)+'.png', 'best_mps_run-' + str(run_num)+'.png',\
                'worst_mps_run-' + str(run_num)+'.png', 'medium_mps_run-' + str(run_num)+'.png']
            figure_filenames = [os.path.join(out_dir, subj_folder, name) for name in figure_names]
            for fig_num, fig in enumerate(figures):
                fig.savefig(figure_filenames[fig_num], dpi=300)
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Decoding model app. Reconstructs modulation power spectrum '\
    'from aligned preprocessed stimulus and fmri data. User specifies input dir, output dir and config file path. '\
    'Config file shall be .json file containing valid arguments for decoding_model_function.')
    parser.add_argument('-inp','--input_data_dir', type=str, help='Path to preprocessed stimuli and fmri')
    parser.add_argument('-out','--output_dir', type=str, help='Path to the output directory where the model \
    data shall be saved. Folder structure can be pre-created or absent')
    parser.add_argument('-config','--config_file', type=str, help='Path to json config file \
    containing subject list, run list, alphas, do_pca_frmi, var_explained and key-value arguments for \
    decoding_model function. IT SI RECOMMENDED TO STORE CONFIG FILE IN THE OUTPUT DIR.') 
    args = parser.parse_args()
    
    with open(args.config_file) as conf:
        config = json.load(conf)   
    # set the defaults in arguments are not specified in config
    if 'subjects' in config: 
        subjects = config['subjects'] 
    else:
        subjects =['01','02','03','04','05','06','09','10','14','15','16','17','18','19','20']
    
    if 'runs' in config:
        runs = config['runs']
    else:
        runs = [[1,2,3,4,5,6,7,8] for el in range(len(subjects))]
    
    if 'scorers' in config:
        scorers = config['scorers']
    else:
        scorers = ['product_moment_corr','mean_squared_error']
    
    if 'do_pca_fmri' in config:
        do_pca_fmri = config['do_pca_fmri']
        var_explained = config['var_explained']
    else: 
        do_pca_fmri = False
        var_explained = None 
    stim_param_dir = config['stim_param_dir']
    alpha_end = config['alphas'][0]
    alpha_dense = config['alphas'][1]
    alphas = np.logspace(0, alpha_end, alpha_dense).tolist()
    
    # RUN DECODING MODEL WITH THE GIVEN ARGUMENTS   
    decoding_model(args.input_data_dir, args.output_dir, stim_param_dir, subjects, runs, scorers, alphas,\
        do_pca_fmri, var_explained)


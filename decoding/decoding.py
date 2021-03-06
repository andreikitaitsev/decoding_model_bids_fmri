#! /usr/bin/env python3
'Decoding model with script_based approach'

### Dependencies
import numpy as np
import copy
import numpy as np
import matplotlib.pyplot as plt
from encoding import product_moment_corr, ridge_gridsearch_per_target
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
import sys
import os
import json
import joblib

__all__ = ['reduce_dimensionality', 'compute_correlation','plot_score_across_features',\
    'plot_score_across_mps', 'reshape_mps', 'plot_mps_and_reconstructed_mps', 'assess_predictions',\
    'denormalize','params_are_equal', 'stack_runs', 'invoke_decoders', 'lag','myRidge',\
    'temporal_decoder', 'decode', 'run_decoding']
### Helper functions

# decorator
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        tic = time.time()
        func(*args, **kwargs)
        toc = time.time() - tic
        print('Elapsed time '+str(toc/60)+' minutes.')
    return wrapper

def reduce_dimensionality(X, var_explained, examine_variance=False, **kwargs):
    ''' Function applys pca with user specified variance explained to the input data
    Inputs:
        X - 2d numpy array (samples/voxels)
        var_explained - float, variabce explained
        examine_variance - logical, whether to run function in interactive mode to manually determine number 
        of components to leave by variance explained plot. Default =Fasle
        kwargs - kwargs for sklearn pca obejct 
    Outputs:
        X_backproj - reduced rand data
        pca - pca object trained (fitted) on the supplied data
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
        fig.suptitle('Scree plot of variance explained')
        ax.set_xlabel('Component number')
        ax.set_ylabel('Variance explained, %')
        plt.show(block = False)

        comp2leave=int(input('Enter number of components to leave: '))
        pca = PCA(n_components=comp2leave)
        pca.fit(X)
    else:
        comp2leave = vars_exp[vars_exp<var_explained].shape[0]
    pca=PCA(n_components=comp2leave)
    pca.fit(X)
    X_backproj = pca.transform(X)
    return X_backproj, pca, comp2leave

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
    return np.array(correlations)

def plot_score_across_features(score, score_name, mps_time, mps_freqs):
    ''' Function plots any sklearn score across features (1 value per feature over different MPSs) 
        as imshow and returns figure handle
    Inputs:
        score - 2d numpy array of reshaped score
        score_name - str, name of the score used
        mps_time - list of strings of mps_time from stimulus parameters json file (output of wav_files....py)
        mps_freqs - list of strings of mps_freqs from stimulus parameters json file (output of wav_files....py)
    Outputs:
        fig - figure handle '''
    fig, ax = plt.subplots()
    fig.suptitle(score_name + ' across different features')
    im = ax.imshow(score, origin='lower', aspect='auto')
    # transform mps_time and mps_freqs from list of strings to np 1d array to allow indexing
    mps_time = np.array([float(el) for el in mps_time])
    mps_freqs = np.array([float(el) for el in mps_freqs])
    # x labels
    format_str = lambda x: ["{:.1f}".format(el) for el in x]
    x_inds = np.around(np.linspace(0, len(mps_time), 10, endpoint=False)).astype(int)
    ax.set_xticks(x_inds)
    ax.set_xticklabels(format_str(mps_time[x_inds]))
    ax.set_xlabel('modulation/s')
    # y labels
    y_inds = np.around(np.linspace(0, len(mps_freqs), 10, endpoint=False)).astype(int)
    ax.set_yticks(y_inds)
    ax.set_yticklabels(format_str(mps_freqs[y_inds]))
    ax.set_ylabel('modulation/Hz')
    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(score_name)
    return fig

def plot_score_across_mps(score, score_name, ylim = None):
    '''Fucntion plots score across different MPSs (1 value per MPS) as curve with plt.plot.
    Inputs:
        score - 1d numpy array of score (1 value per MPS)
        score_name - str, name of the score
        ylim - list of 2 floats; Default = [0,1]
     Outputs:
        fig - figure handle
     '''
    fig, ax = plt.subplots()
    fig.suptitle(score_name + ' across different MPSs')
    ax.plot(score)
    ax.set_xlabel('MPS index')
    ax.set_ylabel(score_name + ' per MPS')
    if ylim != None:
       ax.set_ylim([ylim[0],ylim[1]])
    elif ylim == None:
        ax.set_ylim([-1,1])
    return fig

def reshape_mps(mps, mps_shape):
    ''' Fucntion reshapes (flattened) MPS array into its original form (mod/s, mod/Hz, n_mps) in accordance with
    mps_shape from stimulus json parameters file
    Inputs:
        mps - 2d numpy array of shape (n_mps, n_features) - output of make_X_Y function for a single wav file
        mps_shape - tuple(mod/s, mod/Hz) - shape of a single MPS stored in stimulus parameters json file
    Outputs:
        reshaped_mps - 3d numpy array of shape (n_mps, mod/s, mod/Hz) - same shape as original MPS
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
        fig - figure handle
    '''
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
    format_str = lambda x: ["{:.1f}".format(el) for el in x]
    x_inds = np.around(np.linspace(0, len(mps_time), 10, endpoint=False)).astype(int)
    ax1.set_xticks(x_inds)
    ax1.set_xticklabels(format_str(mps_time[x_inds]))
    ax1.set_xlabel('modulation/s')
    ax2.set_xticks(x_inds)
    ax2.set_xticklabels(format_str(mps_time[x_inds]))
    ax2.set_xlabel('modulation/s')
    # y labels
    y_inds = np.around(np.linspace(0, len(mps_freqs), 10, endpoint=False)).astype(int)
    ax1.set_yticks(y_inds)
    ax1.set_yticklabels(format_str(mps_freqs[y_inds]))
    ax1.set_ylabel('modulation/Hz')
    ax2.set_yticks(y_inds)
    ax2.set_yticklabels(format_str(mps_freqs[y_inds]))
    ax2.set_ylabel('modulation/Hz')
    # colorbar
    max_value = max([np.amax(original_mps),np.amax(reconstructed_mps)]) 
    min_value = min([np.amin(original_mps), np.amin(reconstructed_mps)])
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('log MPS')
    cbar1.mappable.set_clim(min_value, max_value)
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('log MPS')
    cbar2.mappable.set_clim(min_value, max_value)
    # figure title
    if not fig_title is None:
        fig.suptitle(fig_title)
    return fig


def assess_predictions(orig_stim, predicted_stim, parameters, primary_stim_shapes):
    ''' Function computes user-defined measures of quality of predictions between 
    original and predicted data)
    Inputs:
        orig_stim - 2d numpy array of original stimulusrepresentation
        predicted_stim - 2d numpy array of predicted stimulus representation
        parameters - list of dictionary of stimulus parameters (output of the feature extractor)
                     for different runs
        primary_stim_shapes - list of tuples of stim shapes in different runs before stacking
    Outputs:
        assessments - dictionary of user specified assessment quantities
    '''
    # Note: substitution of parameters for concatenated multirun_stim is with parameters
    # for the first run (parameters[0] is justified as we check that all the parameters are
    # equal between runs beforeehand (in stack-runs function)
    # compute correlation coefficient for every feature across different MPSs and plot it
    r2 = product_moment_corr(orig_stim, predicted_stim) 
    resh_r2 = lambda x: np.reshape(x, ((tuple(map(int, parameters[0]['mps_shape'])))))
    r2 = resh_r2(r2)
    fig_r2 = plot_score_across_features(r2, 'Correlation coefficients', parameters[0]['mps_time'], parameters[0]['mps_freqs']) 
     
    # compute correlation across different MPSs between reconstructed MPSs  and original MPSs and plot it
    correlations = compute_correlation(orig_stim, predicted_stim) 
    fig_cor = plot_score_across_mps(correlations, 'Correlation')
    
    # denormalize original and reconstructed MPS
    denorm_orig_stim = []
    denorm_predicted_stim = []
    splits = [shape[0] for shape in primary_stim_shapes]
    splits = [0] + splits
    splits = np.cumsum(splits)
    for i,j in zip(range(1,len(splits)), range(len(primary_stim_shapes))):
        stim_iter = orig_stim[splits[i-1]:splits[i],:]
        predicted_stim_iter = predicted_stim[splits[i-1]:splits[i],:]
        
        denorm_orig_stim.append(denormalize(stim_iter, parameters[j]))
        denorm_predicted_stim.append(denormalize(predicted_stim_iter, parameters[j]))
    denorm_orig_stim = np.concatenate(denorm_orig_stim, axis=0)
    denorm_predicted_stim = np.concatenate(denorm_predicted_stim, axis=0)
    
    # reshape original and reconstructed MPS
    orig_mps = reshape_mps(denorm_orig_stim, parameters[0]['mps_shape'])
    reconstr_mps = reshape_mps(denorm_predicted_stim, parameters[0]['mps_shape'])

    # plot denormalized original and reconstructed MPS with best, worst and "medium" correlation values 
    best_mps_ind = np.argmax(np.squeeze(np.abs(correlations)))
    best_mps = np.squeeze(reconstr_mps[:,:,best_mps_ind]) 
    fig_mps_best = plot_mps_and_reconstructed_mps(orig_mps[:,:,best_mps_ind], best_mps,\
        parameters[0]['mps_time'], parameters[0]['mps_freqs'], 'Best MPS')

    worst_mps_ind = np.argmin(np.squeeze(np.abs(correlations)))
    worst_mps = np.squeeze(reconstr_mps[:,:,worst_mps_ind]) 
     
    fig_mps_worst = plot_mps_and_reconstructed_mps(orig_mps[:,:,worst_mps_ind], worst_mps,\
        parameters[0]['mps_time'],  parameters[0]['mps_freqs'], 'Worst MPS')

    medium_mps_ind = np.argmin(np.squeeze(np.abs(correlations-np.mean(correlations))))
    medium_mps = np.squeeze(reconstr_mps[:,:,medium_mps_ind])
    fig_mps_medium = plot_mps_and_reconstructed_mps(orig_mps[:,:,medium_mps_ind], medium_mps,\
        parameters[0]['mps_time'], parameters[0]['mps_freqs'], 'Medium MPS')
    
    figs = {"correlations": fig_cor, "R2": fig_r2, "Best_MPS": fig_mps_best,\
        "Worst_MPS": fig_mps_worst, "Medium_MPS": fig_mps_medium}  
    mps_inds = {'best_mps':int(best_mps_ind), 'worst_mps':int(worst_mps_ind), 'medium_mps':int(medium_mps_ind)}
    return {"correlations": correlations, "r2": r2, "figs": figs, "mps_inds": mps_inds}

def denormalize(stim, parameters):
    ''' Function adds mean and SD to the flattened stimulus representation by reading 
    it from stim_param file (output of feature extractor)
    Inputs:
        stim - 2d numpy array of flattened stimulus representation (time, MPSs)
        parameters - python dictionary (output of the feature extractor)
    Output:
        rescaled_stim - signal mutliplied by SD and incremented with mean of 
                          original signal
    '''
    mean = np.array(parameters["mps_mean"])
    sd = np.array(parameters["mps_sd"])
    if stim.shape[1] != mean.shape[0] or stim.shape[1] != sd.shape[0]:
        raise ValueError("The sizes of mean and sd arrays do not match the size\
            of the stimulus")
    stim = stim*sd + mean
    return stim

def params_are_equal(params_list):
    '''Function to check if all the parameters of feature extractor 
    in different runs of a signle subject are equal.
    Inputs:
        params - list of dictionaries (stim_param) - outputs of the feature 
                 extractor for each run
    Outputs:
        params_equal - logical, True if all the parameters are equal for every
                       dict in params list
    '''
    params_equal = True
    for run_num in range(0, len(params_list)-1):
        for key in params_list[run_num].keys():
            if not params_list[run_num][str(key)] == params_list[run_num+1][str(key)] and \
                key != 'mps_mean' and key != 'mps_sd':
                params_equal = False
                break
    return params_equal

def stack_runs(fmri_runs, stim_runs, stim_param_list, subj_num):
    '''Function stacks fmri and stim from different runs into 2 2d numpy arrays
    and checks that stim parameters of samples are consistent between runs'''

    # make sure stim parameters are the same for all runs    
    if not params_are_equal(stim_param_list):
        raise ValueError("Stimuli parameters are not equal between runs for subject "+\
        str(subj_num) + "!")
    
    # crop all run voxels to fit minimal amount of voxels across runs
    min_nvox = min([run.shape[1] for run in fmri_runs])
    fmri_runs_cropped = [run[:,:min_nvox] for run in fmri_runs]
    
    # concatenate multirun data
    fmri_multirun = np.concatenate(fmri_runs_cropped, axis=0)
    stim_multirun = np.concatenate(stim_runs, axis=0)
    return fmri_multirun, stim_multirun

def invoke_decoders(decoders, decoders_configs):
    '''Function creates decoder objects and configures them.
    Inputs:
        decoders - list of strings, decoder names
        decoders_configs - list of dictionaries - decoder-specific configs
    Outputs:
        invoked_decoders - list of configured decoder objects
    '''
    invoked_decoders = []
    for dec_num in range(len(decoders)):
        decoder = decoders[dec_num]
        try:
            decoder = eval(decoder)
        except:
            if not isinstance(decoder, str):
                pass
            if isinstance(decoder, str):
                raise ValueError("There is no such decoder as "+str(decoder)+"!")
        decoder = decoder(**decoders_configs[dec_num])
        invoked_decoders.append(decoder)
    return invoked_decoders

def lag(X, lag_par):
    '''
    Inputs:
        X -2d numpy array
        lag_par - int
    Outputs:
        X_lagged - 2d numpy array
    '''
    X_lagged = []
    for i in range(0, lag_par+1):
        X_lagged.append( np.vstack((X[i:,:], np.zeros((i, X.shape[1]))))) 
    return np.concatenate(X_lagged, axis=1)

### Custom decoder classes

class myRidge(BaseEstimator):
    def __init__(self, alphas = (0,5,5), voxel_selection=True, var_explained=None, n_splits = 8, n_splits_gridsearch = 5, normalize=True):
        '''
        kwargs - additional arguments transferred to ridge_gridsearch_per_target
        alphas - list of 3 int (start, stop, num parameters in numpy logspace function),
                 optional. Default: (0,5,5)
                 Regularization parameters to be used for Ridge regression
        voxel_selection - bool, optional, default True
                          Whether to only use voxels with variance larger than zero.
        var_explained - float, whether to do pca on frmi with defiend variance explained. 
                        Default = None (no pca)
        n_splits - int, number of splits in cross validation in fit_transform method
        n_splits_gridsearch - int, number of cross-validation splits in ridge_gridsearch_per_target.
        normalize -bool, whether to do enable normalization in ridge regression. Default = True.
        '''
        if isinstance(alphas, tuple) or isinstance(alphas, list) and len(alphas)==3:
            self.alphas = np.logspace(*list(map(int, alphas)))
        else:
            self.alphas = list(map(float,alphas))
        self.voxel_selection = voxel_selection
        self.n_splits_gridsearch = n_splits_gridsearch
        self.n_splits = n_splits
        self.var_explained = var_explained
        self.pca = None
        self.ridge_model = None
        self.normalize=normalize

    def fit(self, fmri, stim):
        '''Fit ridges to stimulus and frmi using k fold cross validation and voxel selection
        fmri - 2d numpy array of bold fmri data of shape (samples, voxels)
        stim - 2d numpy array of stimulus representation of shape (samples, features)
        '''
        if self.voxel_selection:
            voxel_var = np.var(fmri, axis=0)
            fmri = fmri[:,voxel_var > 0.]
        if self.var_explained != None:
            # return pca object and keep it
            fmri, pca, _ = reduce_dimensionality(fmri, self.var_explained)
            self.pca = pca
        self.ridge_model = ridge_gridsearch_per_target(fmri, stim, self.alphas, n_splits = \
            self.n_splits_gridsearch, normalize=self.normalize)
    
    def predict(self, fmri):
        ''' Returns the stimulus predicted from model trained on test set'''
        if self.var_explained is not None:
            return self.ridge_model.predict(self.pca.transform(fmri))
        else:
            return self.ridge_model.predict(fmri)
    
    def transform(self, fmri):
        '''To use this call with skelarn Pipeline the class shall have transform 
        method. In this setup it is the same as predict.'''
        if self.var_explained is not None:
            return self.ridge_model.predict(self.pca.transform(fmri))
        else:
            return self.ridge_model.predict(fmri)

    def fit_transform(self, fmri, stim):
        '''Method is written specifically for sklearn Pipeline.
        It uses n_fold cross-validation to predict out of 
        sample data X_new after being trained on X'''
        pred_data = []
        kfold = KFold(n_splits = self.n_splits)
        for train, test in kfold.split(fmri, stim):
            self.fit(fmri[train,:], stim[train,:])
            pred_data.append(self.predict(fmri[test,:]))
        pred_data = np.concatenate(pred_data, axis=0)
        self.fit(fmri,stim)
        return pred_data


class temporal_decoder(BaseEstimator):
    '''Class converts any object with .fit and .predict methods into
    final (temporal) decoder for spatial temporal model.''' 
    def __init__(self, decoder, decoder_config, lag_par):
        super().__init__() 
        self.lag_par = lag_par
        self.decoder = decoder(**decoder_config)
    def fit(self, x, y):
        '''Somewhy when used with CCA as spatial decoder receives
        (x_tr,y_tr) and y_orig as inputs.'''
        if (isinstance(x, tuple) or isinstance(x,list)) and len(x) ==2:
            x_tr=x[0]
            self.decoder.fit(lag(x_tr, self.lag_par), y)
        else:
            self.decoder.fit(lag(x, self.lag_par), y)
    def predict(self, x):
        '''In sklearn pipeline the predict method of the last estimator is
        called on sequence of outputs of .transform (or .fit_transform) methods of
        previous estimators. Some (supervised) estimators return X_tr as output
        of .transform (e.g. PCA), some X_tr, Y_tr (e.g. CCA). Therefore we take 
        only first (X) element of the input (X_tr, Y_tr) to .predict method 
        of temporal decoder if the input to .predict is a tuple or list.'''
        if (isinstance(x, tuple) or isinstance(x, list)) and len(x) == 2:
            x_ = x[0]
        else:
            x_ = x
        return self.decoder.predict(lag(x_, self.lag_par))



### Decoding functions

def decode(fmri, stim, decoder, n_splits=8):
    '''Runs decoding one stimulus and fmri runi
    Inputs:
        fmri - 2d numpy array of preprocessed fmri
        stim - 2d numpy array of  preprocessed stim
        decoder - decoder object with .fit and .predict methods
        n_splits - int, number of cross-validation splits;
        Default = 8.
    Outputs:
        predictions - 2d numpy array of predicted stimulus representation
    '''
    # preallocate outputs
    decoders = []
    predictions = []
    # create cross-validation object
    kfold = KFold(n_splits=n_splits)
    #predictions=cross_val_predict(decoder, fmri, stim, cv=kfold, n_jobs=n_splits)
     
    for train, test in kfold.split(fmri, stim):
        # copy decoder object
        dec = copy.deepcopy(decoder)
        decoders.append(dec)
        # fit a copy of decoder object on train split
        decoders[-1].fit(fmri[train,:], stim[train,:])
        # predict stimulus from trained object
        predictions.append(decoders[-1].predict(fmri[test,:]))
        # concatenate predictions
    predictions = np.concatenate(predictions, axis=0)
    return predictions

@timer
def run_decoding(inp_data_dir, out_dir, stim_param_dir, model_config, decoder):
    '''Runs decoding model for user-specified subjects and runs and saves precitions
    and assessments data into the output dir.
    Inputs:
        inp_data_dir - str, input directory with preprocessed BOLD and stimrepresentation
        out_dir - str, directory where predictions and assessment data will be saved into
        stim_param_dir - str, directory where parameters of the feature extractor are stored
        model_config - python dictionary with subjects as list of strings, runs as list of int or str
                       and multirun keys. If multirun is false, model does single bold file decoding. 
                       Default=True
        decoder - any object ingeriting from sklearn BaseEstimator class, implementing .fit
                  and .predict methods.
    '''
    ### Input check (as model takes long time to run (would a be pity to find a glitch after
    #running model for days:) )
    
    # unpack subjects and runs
    if "subjects"in model_config:
        subjects = model_config["subjects"]
    else:
        subjects =['01','02','03','04','05','06','09','14','15','16','17','18','19','20']
    
    if "runs" in model_config:
        runs = model_config["runs"]
    else:
        runs = [[1,2,3,4,5,6,7,8] for el in range(len(subjects))]
    
    if "multirun" not in model_config:
        model_config["multirun"] = True

    if len(subjects) != len(runs):
        raise ValueError("Number of subjects does not match number of runs you have specified!"
        " Subjects shall be list of subject numbers and runs shall be list of lists of run numbers"
        "for every subject you have specified.")
    
    # loop through subject folders and runs and make sure all the files exist 
    for subj_counter, subj_num in enumerate(subjects):
        subj_folder = 'sub-'+ str(subj_num) 
        subj_folder_path = os.path.join(inp_data_dir, subj_folder)
        if not os.path.isdir(subj_folder_path):
            raise FileNotFoundError("Directory "+subj_folder_path+" does not exist!")
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
            os.makedirs(os.path.join(out_dir, subj_folder))
        elif len(os.listdir(os.path.join(out_dir, subj_folder))) != 0:
            proceed = input('Directory ' + os.path.join(out_dir, subj_folder) + ' is not empty!\n Proceed? y/n \n')
            if proceed == 'n':
                return None
            elif proceed =='y':
                pass


    for subj_counter, subj_num in enumerate(subjects):
        subj_folder = 'sub-'+ str(subj_num) 
        subj_folder_path = os.path.join(inp_data_dir, subj_folder)
        
        ### sungle file decoding
        if model_config["multirun"] == False:
            fmri_fname = subj_folder + '_task-aomovie' + '_bold.tsv.gz'
            stim_fname = 'task-aomovie' + '_stim.tsv.gz'
            stim_param = 'task-aomovie' + '_stim_parameters.json'
            fmri_path = os.path.join(subj_folder_path, fmri_fname)
            stim_path = os.path.join(subj_folder_path, stim_fname)
            stim_param_path = os.path.join(stim_param_dir, stim_param)
            
            # Load data
            with open(fmri_path, 'rb') as fm:
                fmri = joblib.load(fm) 
            if not np.ndim(fmri_iter) == 2:
                raise ValueError('Inputs shall be 2d numpy array' 
                ' fmri shape was'+ str(fmri.shape)+ 'instead.')
            with open(stim_path, 'rb') as st:
               stim = joblib.load(st)
            if not np.ndim(stim) == 2:
                raise ValueError('Inputs shall be 2d numpy array' 
                        ' stimulus shape was'+ str(stim.shape)+ 'instead.')
            with open(stim_param_path, 'r') as par:
                parameters = [json.load(par)]
            orig_stim_shapes=[stim.shape]

        ### multirun_decoding (default - mutlirun)
        elif model_config["multirun"]: 
            fmri = []
            stim = []
            parameters = []
            orig_stim_shapes = [] 
            
            for run_num in runs[subj_counter]:
                # get fmri, stimulus and stim_parameters paths 
                fmri_fname = subj_folder + '_task-aomovie_run-' + str(run_num) + '_bold.tsv.gz'
                stim_fname = 'task-aomovie_run-' + str(run_num) + '_stim.tsv.gz'
                stim_param = 'task-aomovie_run-' + str(run_num) + '_stim_parameters.json'
                fmri_path = os.path.join(subj_folder_path, fmri_fname)
                stim_path = os.path.join(subj_folder_path, stim_fname)
                stim_param_path = os.path.join(stim_param_dir, stim_param)
                
                # Load data
                with open(fmri_path, 'rb') as fm:
                    fmri_iter = joblib.load(fm) 
                if not np.ndim(fmri_iter) == 2:
                    raise ValueError('Inputs shall be 2d numpy array' 
                    ' fmri shape was'+ str(fmri_iter.shape)+ 'instead.')
                with open(stim_path, 'rb') as st:
                   stim_iter = joblib.load(st)
                if not np.ndim(stim_iter) == 2:
                    raise ValueError('Inputs shall be 2d numpy array' 
                            ' stimulus shape was'+ str(stim_iter.shape)+ 'instead.')
                with open(stim_param_path, 'r') as par:
                    param_iter = json.load(par)
                
                # Append runs
                fmri.append(fmri_iter)
                stim.append(stim_iter)
                parameters.append(param_iter)
                orig_stim_shapes.append(stim_iter.shape)
            
            # get concatenated multirun datasets
            fmri, stim = stack_runs(fmri, stim, parameters, subj_num)
        
        ## run decoding
        predictions = decode(fmri, stim, decoder)     
        
        # assess predictions
        assessments = assess_predictions(stim, predictions, parameters, orig_stim_shapes)
        
        # save model data into subject-specific folder in the out_dir
        print('saving model data for subject ', str(subjects[subj_counter]) + "...")
        predictions_name = 'reconstructed_mps_' + str(subj_folder) + '.pkl'
        joblib.dump(predictions, os.path.join(out_dir, subj_folder, predictions_name))

        # save assessment data
        for assessment in assessments.keys():
            filename = assessment + '_' + str(subj_folder) + '.pkl'
            joblib.dump(assessments[assessment], os.path.join(out_dir, subj_folder, filename))

        # save figures (fig_cor, fig_mse, fig_mps_best, fig_mps_worst, fig_mps_medium) 
        for fig in assessments["figs"].keys():
            fig_name = fig + '_' + str(subj_folder) + '.png'
            fig_filename = os.path.join(out_dir, subj_folder, fig_name)
            assessments["figs"][fig].savefig(fig_filename, dpi = 300)


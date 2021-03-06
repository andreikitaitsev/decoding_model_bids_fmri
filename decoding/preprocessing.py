import os
import warnings
import numpy as np
import joblib
from nilearn.masking import unmask, apply_mask
from nibabel import save, load, Nifti1Image
from nilearn.signal import clean

__all__ = ['preprocess_bold_fmri', 'get_remove_idx', 'make_X_Y']

def preprocess_bold_fmri(bold, mask=None, detrend=True, standardize='zscore', **kwargs):
    '''Preprocesses data and returns ndarray.'''
    if mask:
        data = apply_mask(bold, mask)
    else:
        if not isinstance(bold, Nifti1Image): 
            data = load(bold).get_data()
        else:
            data = bold.get_data()
        data = np.reshape(data, (-1, data.shape[-1])).T
    return clean(data, detrend=detrend, standardize=standardize, **kwargs)

def get_remove_idx(lagged_stimulus, remove_nan=True):
    '''Returns indices of rows in lagged_stimulus to remove'''
    if remove_nan is True:
        return np.where(np.any(np.isnan(lagged_stimulus), axis=1))[0]
    elif remove_nan <= 1. and remove_nan >= 0.:
        return np.where(np.isnan(lagged_stimulus).mean(axis=1) > remove_nan)[0]
    else:
        raise ValueError('remove_nan needs to be either True, False, or a float between 0 and 1.')


def generate_lagged_stimulus(stimulus, fmri_samples, TR, stim_TR,
                             lag_time=6.0, start_time=0., offset_stim=0.,
                             fill_value=np.nan):
    '''Generates a lagged stimulus representation temporally aligned with the fMRI data

    Parameters
    ----------
    stimuli : ndarray, stimulus representation of shape (samples, features)
    fmri_samples : int, samples of corresponding fmri run
    TR : int, float, repetition time of the fMRI data in seconds
    stim_TR : int, float, repetition time of the stimulus in seconds
    lag_time : int, float, or None, optional,
               lag to introduce for stimuli in seconds,
               if no lagging should be done set this to TR or None
    start_time :  int, float, optional, default 0.
                  starting time of the stimulus relative to fMRI recordings in seconds
                  appends fill_value to stimulus representation to match fMRI and stimulus
    offset_stim : int, float, optional, default 0.
                  time to offset stimulus relative to fMRI in the lagged stimulus,
                  i.e. when predicting fmri at time t use only stimulus features
                  before t-offset_stim. This reduces the number of time points used
                  in the model.
    fill_value : int, float, or any valid numpy array element, optional, default np.nan
                 appends fill_value to stimulus array to account for starting_time
                 use np.nan here with remove_nans=True to remove fmri/stimulus samples where no stimulus was presented

    Returns
    -------
    ndarray of the lagged stimulus of shape (samples, lagged features)
    '''
    from skimage.util import view_as_windows
    # find out temporal alignment
    stim_samples_per_TR = TR / stim_TR
    if stim_samples_per_TR < 1:
        raise ValueError('Stimulus TR is larger than fMRI TR')
    # check if result is close to an integer
    if not np.isclose(stim_samples_per_TR, np.round(stim_samples_per_TR)):
        warnings.warn('Stimulus timing and fMRI timing do not align. '
        'Stimulus samples per fMRI samples: {0} for stimulus TR {1} and fMRI TR {2}. '
        'Proceeds by rounding stimulus samples '
        'per TR.'.format(stim_samples_per_TR, stim_TR, TR), RuntimeWarning)
    stim_samples_per_TR = int(np.round(stim_samples_per_TR))
    if lag_time is None:
        lag_time = TR
    # check if lag time is multiple of TR
    if not np.isclose(lag_time / TR, np.round(lag_time / TR)):
        raise ValueError('lag_time should be a multiple of TR so '
                'that stimulus/fMRI alignment does not change.')
    if lag_time == TR:
            warnings.warn('lag_time is equal to TR, no stimulus lagging will be done.', RuntimeWarning)
    lag_TR = int(np.round(lag_time / TR))
    offset_TR = int(np.round(offset_stim / TR))

    n_features = stimulus.shape[1]
    n_append = 0
    n_prepend = 0
    # check if the stimulus start time is moved w.r.t. fmri
    n_prepend += int(np.round(start_time / stim_TR))
    stimulus = np.vstack([np.full((n_prepend, n_features), fill_value), stimulus])

    # make reshapeable by appending filler
    if stimulus.shape[0] % stim_samples_per_TR > 0:
        # either remove part of the stimulus (if it is longer than fmri) or append filler
        if stimulus.shape[0] / stim_samples_per_TR > fmri_samples:
            stimulus = stimulus[:-(stimulus.shape[0] % stim_samples_per_TR)]
        else:
            n_append = stim_samples_per_TR - ((stimulus.shape[0]) % stim_samples_per_TR)
            stimulus = np.vstack([stimulus, np.full((n_append, n_features), fill_value)])

    # now reshape and lag
    # TODO: check for memory footprint wrt copying
    stimulus = np.reshape(stimulus, (-1, stim_samples_per_TR * n_features))

    # check if stimulus is longer than fmri and remove part of the stimulus
    if stimulus.shape[0] > fmri_samples:
        warnings.warn('Stimulus ({0}) is longer than recorded fMRI '
                      '({1}). Removing last part of stimulus.'.format(stimulus.shape[0]*TR, fmri_samples*TR))
        stimulus = stimulus[:fmri_samples]


    # check if lagging should be done
    if lag_time != TR:
        # account for lagging
        n_prepend_lag = (lag_TR + offset_TR) - 1
        # and add filler such that length is the same for fmri
        n_append_lag = fmri_samples - stimulus.shape[0]
        stimulus = np.vstack(
                             [np.full((n_prepend_lag, n_features * stim_samples_per_TR), fill_value),
                              stimulus,
                              np.full((n_append_lag, n_features * stim_samples_per_TR), fill_value)])
        stimulus = np.swapaxes(np.squeeze(view_as_windows(stimulus, ((lag_TR + offset_TR), 1)) ), 1, 2)
        stimulus = np.reshape(stimulus, (stimulus.shape[0], -1))

    # remove stimulus representations that are more recent than offset_stim
    if offset_stim > 0:
        stimulus = stimulus[:, :-(offset_TR *stim_samples_per_TR * n_features)]
    return stimulus

def make_X_Y(stimuli, fmri, TR, stim_TR, lag_time=6.0, start_times=None, offset_stim=0., fill_value=np.nan, remove_nans=True):
    '''Creates (lagged) features and fMRI matrices concatenated along runs

    Parameters
    ----------
    stimuli : list, list of stimulus representations
    fmri : list, list of fMRI ndarrays
    TR : int, float, repetition time of the fMRI data in seconds
    stim_TR : int, float, repetition time of the stimulus in seconds
    lag_time : int, float, optional,
               lag to introduce for stimuli in seconds,
               if no lagging should be done set this to TR
    start_times : list, list of int, float, optional,
                  starting time of the stimuli relative to fMRI recordings in seconds
                  appends fill_value to stimulus representation to match fMRI and stimulus
    offset_stim : int, float, optional,
                  time to offset stimulus relative to fMRI in the lagged stimulus,
                  i.e. when predicting fmri at time t use only stimulus features
                  before t-offset_stim. This reduces the number of time points used
                  in the model.
    fill_value : int, float, or any valid numpy array element, optional,
                 appends fill_value to stimulus array to account for starting_time
                 use np.nan here with remove_nans=True to remove fmri/stimulus samples where no stimulus was presented
    remove_nans : bool, bool or float 0<=remove_nans<=1, optional
                  True/False indicate whether to remove all or none
                  stimulus/fmri samples that contain nans
                  a proportion keeps all samples in the lagged stimulus that have
                  lower number of nans than this proportion.
                  Replace nans with zeros in this case.
    
    Returns
    -------
    tuple of two ndarrays
    the first element are the (lagged) stimuli
    the second element is the aligned fMRI data

    Examples
    --------
    >>> stim_TR, TR = 0.1, 2
    >>> stimulus = np.tile(np.arange(80)[:, None], (1, 1))
    >>> fmri = np.tile(np.arange(0, 4)[:, None], (1, 1))
    >>> make_X_Y([stimulus], [fmri], TR, stim_TR, lag_time=4, offset_stim=0, start_times=[0])
    (array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
            13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
            26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
            39.],
           [20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
            33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45.,
            46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58.,
            59.],
           [40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52.,
            53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
            66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76., 77., 78.,
            79.]]), array([[1],
           [2],
           [3]]))
    '''
    from skimage.util import view_as_windows
    if len(stimuli) != len(fmri):
        raise ValueError('Stimulus and fMRI need to have the same number of runs. '
        'Instead fMRI has {} and stimulus {} runs.'.format(len(fmri), len(stimuli)))
    n_features = stimuli[0].shape[1]
    if not np.all(np.array([stim.shape[1] for stim in stimuli]) == n_features):
        raise ValueError('Stimulus has different number of features per run.')

    lagged_stimuli = []
    aligned_fmri = []
    for i, (stimulus, fmri_run) in enumerate(zip(stimuli, fmri)):
        stimulus = generate_lagged_stimulus(
            stimulus, fmri_run.shape[0], TR, stim_TR, lag_time=lag_time,
            start_time=start_times[i] if start_times else 0.,
            offset_stim=offset_stim, fill_value=fill_value)
        # remove nans in stim/fmri here
        if remove_nans:
            remove_idx = get_remove_idx(stimulus, remove_nans)
            stimulus = np.delete(stimulus, remove_idx, axis=0)
            fmri_run = np.delete(fmri_run, remove_idx, axis=0)

        # remove fmri samples recorded after stimulus has ended
        if fmri_run.shape[0] != stimulus.shape[0]:
            warnings.warn('fMRI data and stimulus samples differ. '
            'Removing additional fMRI samples. This could mean that you recorded '
            'long after stimulus ended or that something went wrong in the '
            'preprocessing. fMRI: {}s stimulus: {}s'.format(
                TR*fmri_run.shape[0], TR*stimulus.shape[0]), RuntimeWarning)
            if fmri_run.shape[0] > stimulus.shape[0]:
                fmri_run = fmri_run[:-(fmri_run.shape[0]-stimulus.shape[0])]
        lagged_stimuli.append(stimulus)
        aligned_fmri.append(fmri_run)
    return np.vstack(lagged_stimuli), np.vstack(aligned_fmri)


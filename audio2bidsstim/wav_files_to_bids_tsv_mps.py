import numpy as np
import joblib
import glob
import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import librosa as lbr
import json
import os
import warnings            
import pandas as pd 
__all__ = ['standardize_mps', 'mps_stft','get_mel_spectrogram']

def standardize_mps(mps, return_mean_and_sd=False):
    '''Function standardizes flattened MPS of shape (times, features) by dividing each feature into its 
    standard deviation across times and subtracting the mean across times. Function can also return 
    the original mean and SD as outputs
    Inputs:
        mps                - 2d numpy array of shape (times, features)
        return_mean_and_sd - logical, whether to return the mean and SD of original MPS as outputs
                             Default = False
    Output:
        std_mps - 2d numpy array of shape (times, features) - standardized MPS
        mean    - 1d numpy arrays - the mean of the original MPS
        sd      - 1d numpy array - SD of the original MPS
    ''' 
    mean = np.mean(mps, axis=0)
    sd = np.std(mps, axis = 0)
    std_mps = np.divide((mps-mean), np.std(mps, axis = 0))
    if return_mean_and_sd:
        return std_mps, mean, sd
    else:
        return std_mps

def mps_stft(filepath, sr, n_fft_stft, hop_length_stft, n_fft_mps, hop_length_mps, use_power = True, log=True,\
             dB=False, plot_spectr=False, plot_mps=False, return_figures=False, cutoff_temp_mod = 50, cutoff_spectr_mod = 50, dec = 2, **kwargs):
    ''' Function to create modulation powermspectra from wav file via 2d FFT of STFT spectrogram.
    Function also saves its configuration to the output directory as json file
    
    Inputs:
        
    filepath -          path to the wav file
    sr -                sampling rate of wav file
    hop_length_stft -   step size for librosa stft
    n_fft_stft -        window size of stft
    n_fft_mps -         window size for 2d Fourier transform of spectrogram (along the time axis)
    hop_length_mps -    step size of 2d Fourier transform of spectrogram along the time axis
    use_power -         use power spectrum instead of amplitude (ampl^2). (default = True)
    log -               whether to use logarithm of spectrogram for mps calculation (default = True)
    dB -                flag whether to use amplitude or power spectrum for 2d FFT (default = False)
    plot_spectr -       flag, whether to plot spectrogram or not (default = False)
    plot_mps -          flag, whether to plot mps of 1 time scliece (default  = False)
    dec -               number of non-zero decimals in feature names
    return_figures -       flag, whether to return figure handlers (if there are any) 
    cutoff_temp_mod -   cutoff of modulationd/s (Hz). (default = 50)
    cutoff_spectr_mod - cutoff of modulation/Hz. (default = 50)
    kwargs -            key-value arguements to librosa stft function when creating spectrogram or \
                        numpy fft2 fucntion when creating MPS
    Note1: log and dB cannot be True at the same time!
    
    Outputs:
        
    mod_pow_spectrs - modulation power spectra - feature representation (time x features) - numpy 2d array
    params          - python dictionary of fucntion parameters(in the subdict metadata), \
                      feature_names and repetition time in seconds
    metadata        - dictionary, data required according to the bids standard \
                      (repetition time and feature names)
    
    Note, that feature_names - list of strings - all modulatation/s for each modulation/Hz 
    (all mod/s for  mod/Hz==1, all mod/s for md/Hz==2, etc.) 
    repetition time in seconds - float; amount of seconds between 2 mps (2 rows in feature_representation)
    
    '''
    
    if hop_length_mps > n_fft_mps:
        raise ValueError('step size (Hop_length) shall not exceed length of fft window (n_fft_mps)')
    if dB and log:
        raise ValueError("Cannot take the log of decibels (they are often negative). Both dB and log cannot be True.")
    if use_power and dB:
       warnings.warn("You are converting power (not amplitude spectrum) to decibels! \
                     MPS can be discontinuous", RuntimeWarning) 
    
    ## Obtain spectrogram
    soundfile, sr = lbr.load(filepath,sr=sr, mono=True)   
    
    Nyquist = int(np.ceil(n_fft_stft/2))  
    spectrogram = lbr.stft(soundfile, n_fft=n_fft_stft, hop_length=hop_length_stft,\
                           **{arg: kwargs['arg'] for arg in ['win_length', 'window', 'center','dtype','pad_mode'] if arg\
                              in kwargs}) 
    spectrogram = np.abs(spectrogram) # get real-valued signal
    spectrogram = spectrogram[:Nyquist,:] # make sure not to exceed Nyquist; 
    
    ## See that  n_fft_mps and hop_length_mps do not exceed spectrogram length
    if n_fft_mps>=spectrogram.shape[1]: 
        raise ValueError('length of fft window for modulation power spectrum exceeds length of spectrogram')
    
    if hop_length_mps >= spectrogram.shape[1]:
        raise ValueError('step size of fft window (hop_length_mps) for modulation power spectrum exceeds length of spectrogram ')
    
    # calculate stepsize on frequency and time axes of spectrogram
    step_size_spec_time = hop_length_stft/sr #sec/sample
    step_size_spec_freq =  sr/n_fft_stft # Hz/sample - fundamental frequency
    
    
    ## Transform spectrogram in accordance with input parametres
    if use_power:
       spectrogram = np.power(spectrogram, 2)
    if dB and not use_power:
        spectrogram=lbr.amplitude_to_db(spectrogram, ref=np.max)
    if dB and use_power:
        spectrogram = lbr.core.power_to_db(spectrogram, ref=np.max)
    if log: 
       spectrogram = np.log1p(spectrogram)    

    ## Calculate MPS
    
    # Preallocate variables
    mps_accum = []
    mps_iter = []
    mod_pow_spectrs = []
    
    # Step sizes of MPS
    step_size_mps_time = hop_length_mps * step_size_spec_time
    step_size_mps_freq = step_size_spec_freq

    fft_window = np.linspace(0,n_fft_mps-1,n_fft_mps,endpoint=True)
    hop_list = [hop for hop in range(0,spectrogram.shape[1]//n_fft_mps)]
    Nyquist_mps_freq = int(np.ceil(spectrogram.shape[0]/2)) # Nyquist on modulation/Hz axis
    
    for iter in hop_list:
        idx_ar = fft_window + hop_length_mps*iter
        mps_iter = np.abs(np.fft.fftshift( (np.fft.fft2(spectrogram[:,idx_ar.astype(int)], \
                                      **{argnt: kwargs["argnt"] for argnt in ['s','axes','norm'] if argnt in kwargs})) ) )
        
        # get axes units for MPS
        mps_time = np.fft.fftshift( np.fft.fftfreq(fft_window.shape[0], d = step_size_spec_time) ) # modulation/ s
        mps_freqs = np.fft.fftshift( np.fft.fftfreq(mps_iter.shape[0], d = 1/ step_size_spec_freq) )# modulation/ Hz

        # reject mirrored frequencies on Y axis (modulations/Hz) 
        mps_iter = mps_iter[Nyquist_mps_freq:, :]  
        mps_freqs = mps_freqs[Nyquist_mps_freq:]

        # cut the MPS according to the input parameters
        mps_iter = mps_iter[np.where(mps_freqs <= cutoff_spectr_mod)[0],:]
        mps_iter = mps_iter[:, np.where(np.abs(mps_time) <= cutoff_temp_mod)[0] ]        
        
        # update mps_accum later used for plotting
        mps_accum.append(mps_iter)
        
        # flatten 
        mps_iter= np.reshape(mps_iter,(1,mps_iter.shape[0]*mps_iter.shape[1]))
        mod_pow_spectrs.append(mps_iter)
    mod_pow_spectrs = np.concatenate(mod_pow_spectrs, axis=0)
    
    # standardize flattened MPSs across times (subtract mean and sd across times)
    mod_pow_spectrs, mps_mean, mps_sd = standardize_mps(mod_pow_spectrs, return_mean_and_sd = True)

    # check if dec is smaller than step_size_mps_freq 
    if np.round(np.abs(mps_freqs[1] - mps_freqs[0]), decimals = dec) == 0:
        warn_message_freq="The amount of decimals to round feature names you have chosen"+ \
            "is less, than step size of mod/Hz axis. mod/Hz feature names will be aliased" +\
                          "\n number of decimals you chose: " + str(dec) + \
                             "\n step size mod/Hz: " + str(step_size_mps_freq)
        warnings.warn(warn_message_freq, RuntimeWarning())
    # check if dec is smaller than step_size_mps_time
    if np.round(np.abs(mps_time[1] - mps_time[0]), decimals = dec) == 0:
        warn_message_time = "The amount of decimals to round feature names you have chosen" + \
            "is less, than step size of mod/s axis. mod/s feature names will be aliased"+  \
                          "\n number of decimals you chose: " + str(dec) +\
                              "\n step size mod/Hz: " + str(step_size_mps_time)
        warnings.warn(warn_message_time, RuntimeWarning())    
    
    ## create feature names array        
    mps_freqs_crop = mps_freqs[ np.where(mps_freqs <= cutoff_spectr_mod)[0] ]
    mps_time_crop = mps_time [ np.where(np.abs(mps_time) <= cutoff_temp_mod)[0] ]
    feature_names = [ '{0:.{1}f}'.format(mod_freq, dec) for mod_freq in mps_freqs_crop   ]
    # after this line feature_names is mps_freq[0]+mps_time[0:-1],i.e. list of mps_time for every mps_freq
    feature_names = ['{1},{0:.{2}f}'.format(mod_time, mod_freq, dec)\
            for mod_freq in feature_names for mod_time in mps_time_crop]
       

    ## plotting  
    # plot spectrogram- the same window as for 1st mps plot
    if plot_spectr:
        # compute time and frequency axes
        time_s =  fft_window * step_size_spec_time
        frequs_hz = np.linspace(0, spectrogram.shape[0], spectrogram.shape[0],endpoint=False) * step_size_mps_freq 
        # plot
        fig1,ax=plt.subplots()
        img = ax.imshow(spectrogram[:,fft_window.astype(int)], origin='lower', aspect='auto')
        ax.set_title('Spectrogram for the 1st MPS window')
        
        xtickinds = np.around(np.linspace(0, n_fft_mps, 10, endpoint = False)).astype(int)
        ax.set_xticks(xtickinds)
        ax.set_xticklabels( [ '{:.1f}'.format(el) for el in time_s[xtickinds] ] )
        
        ytickinds = np.around(np.linspace(0, spectrogram.shape[0], 10, endpoint=False)).astype(int)
        ax.set_yticks(ytickinds)
        ax.set_yticklabels([ '{:.0f}'.format(el) for el in frequs_hz[ytickinds] ])
        
        ax.set_xlabel('Time, s')
        ax.set_ylabel('Frequency, Hz')
        clb = fig1.colorbar(img, ax=ax)
        if log and use_power:
            clb.set_label('log power')
        if log and not use_power:
            clb.set_label('log amplitude')
        elif dB:
            clb.set_label('dB')
        
    # Plot MPS
    if plot_mps:
        fig2, ax = plt.subplots()
        graph=ax.imshow(np.log(mps_accum[0]), origin='lower', aspect='auto') 
        xtickind = np.around(np.linspace(0, len(mps_time), 10, endpoint = False)).astype(int)
        ax.set_xticks(xtickind)
        ax.set_xticklabels( [ '{:.0f}'.format(el) for el in mps_time[xtickind] ])
        
        ytickind = np.around(np.linspace(0, mps_accum[0].shape[0], 10, endpoint = False) ).astype(int)
        ax.set_yticks(ytickind)
        ax.set_yticklabels( [ '{:.0f}'.format(el) for el in mps_freqs[ytickind] ])
        ax.set_title("log of MPS of the 1st window")
        ax.set_xlabel("Modulation/second")
        ax.set_ylabel("Modulation/Hz")
        cbar=fig2.colorbar(graph, ax=ax)
        cbar.set_label("log(MPS)")
        
    ## create dictionary with parametres for json file
    mps_freqs = [str(freq) for freq in mps_freqs]
    mps_time = [str(time) for time in mps_time]
    mps_repetition_time = step_size_mps_time
    parameters = {'sr':sr,'hop_length_stft': hop_length_stft,'n_fft_stft': n_fft_stft, 'n_fft_mps': n_fft_mps, \
                'hop_length_mps':hop_length_mps, 'dB': dB, 'log':log, 'use_power': use_power, \
                'cutoff_temp_mod': cutoff_temp_mod, 'cutoff_spectr_mod': cutoff_spectr_mod, 'mps_shape':mps_accum[0].shape,\
                    'mps_repetition_time': mps_repetition_time, 'mps_freqs':mps_freqs, 'mps_time':mps_time,\
                    'mps_mean': mps_mean.tolist(), 'mps_sd': mps_sd.tolist()}
    metadata = {'mps_repetition_time': mps_repetition_time, "feature_names": feature_names}  
    if return_figures:
        return mod_pow_spectrs, parameters, metadata, fig1, fig2
    elif not return_figures: 
        return mod_pow_spectrs, parameters, metadata 

 

def get_mel_spectrogram(filename, log=True, sr=44100, hop_length=512, **kwargs):
    '''Returns the (log) Mel spectrogram of a given wav file, the sampling rate of that spectrogram and names of the frequencies in the Mel spectrogram

    Parameters
    ----------
    filename : str, path to wav file to be converted
    sr : int, sampling rate for wav file
         if this differs from actual sampling rate in wav it will be resampled
    log : bool, indicates if log mel spectrogram will be returned
    kwargs : additional keyword arguments that will be
             transferred to librosa's melspectrogram function

    Returns
    -------
    a tuple consisting of the Melspectrogram of shape (time, mels), the repetition time in seconds, and the frequencies of the Mel filters in Hertz 
    '''
    wav, _ = lbr.load(filename, sr=sr)
    melspecgrams = lbr.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length,
                                              **kwargs)
    if log:
        melspecgrams[np.isclose(melspecgrams, 0)] = np.finfo(melspecgrams.dtype).eps
        melspecgrams = np.log(melspecgrams)
    log_dict = {True: 'Log ', False: ''}
    freqs = lbr.core.mel_frequencies(
            **{param: kwargs[param] for param in ['n_mels', 'fmin', 'fmax', 'htk']
                if param in kwargs})
    freqs = ['{0:.0f} Hz ({1}Mel)'.format(freq, log_dict[log]) for freq in freqs]
    return melspecgrams.T, sr / hop_length, freqs


if __name__ == '__main__':
    import argparse
    from itertools import cycle
    parser = argparse.ArgumentParser(description='Wav2bids stim converter.')
    parser.add_argument('file', help='Name of file or space separated list of files or glob expression for wav files to be converted.', nargs='+')
    parser.add_argument('-e', '--extractor',help='Type of feature extractor to use. mps or mel.', type=str, default='mps')
    parser.add_argument('-c' ,'--config', help='Path to json file that contains the parameters to librosa\'s melspectrogram function.')
    parser.add_argument('-o', '--output', help='Path to folder where to save tsv and json files, if missing uses current folder.')
    parser.add_argument('-t', '--start-time', help='Start time in seconds relative to first data sample.'
            ' Either a single float (same starting time for all runs) or a list of floats.', nargs='+', type=float, default=0.)
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as fl:
            config = json.load(fl)
    else:
        config = dict()
    if isinstance(args.file, str):
        args.file = [args.file]
    if len(args.file) == 1 and '*' in args.file[0]:
       	args.file = glob.glob(args.file[0])
    if ( isinstance(args.extractor, str) and args.extractor =='mel'):
        if isinstance(args.start_time, float):
            args.start_time = [args.start_time]
        if len(args.start_time) > 1 and (len(args.start_time) != len(args.file)):
            raise ValueError('Number of files and number of start times are unequal. Start time has to be either one element or the same number as number of files.')
        for wav_file, start_time in zip(args.file, cycle(args.start_time)):
            melspec, sr_spec, freqs = get_mel_spectrogram(wav_file, **config)
            tsv_file = os.path.basename(wav_file).split('.')[0] + '.tsv.gz'
            json_file = os.path.basename(wav_file).split('.')[0] + '.json'
            if args.output:
                tsv_file = os.path.join(args.output, tsv_file)
                json_file = os.path.join(args.output, json_file)
                np.savetxt(tsv_file, melspec, delimiter='\t')
                metadata = {'SamplingFrequency': sr_spec, 'StartTime': start_time,'Columns': freqs}
                with open(json_file, 'w+') as fp:
                    json.dump(metadata, fp)
    elif (isinstance(args.extractor, str) and  args.extractor =='mps'):
            markdown = {'0':'1','1':'2','2':'3','3':'4','4':'5','5':'6','6':'7','7':'8'} #keys - stimuli numbers, values - frmi numebrs
            for wav_file in args.file:
                if ("return_figures" in config and config["return_figures"]):
                    mps, params, metadat, fig1, fig2 = mps_stft(wav_file, **config)	
                else:
                    mps, params, metadat = mps_stft(wav_file, **config)
                filenum = markdown[ os.path.basename(wav_file).split('.')[0] ] 
                tsv_file ='task-aomovie_run-'+filenum+ '_stim' + '.tsv.gz'
                json_stim_metadat ='task-aomovie_run-'+filenum+ '_stim_description'+'.json' 
                json_stim_params = 'task-aomovie_run-'+filenum+ '_stim_parameters'+'.json' 
                if ("return_figures" in config and config["return_figures"]):
                    fig1_fl = 'task-aomovie_run-'+ filenum+ '_stim_' +'spectrogram.png'
                    fig1_fl = os.path.join(args.output,fig1_fl)
                    fig2_fl = 'task-aomovie_run-'+filenum+ '_stim_' + 'mps.png'
                    fig2_fl = os.path.join(args.output,fig2_fl)

                if args.output:
                    tsv_file = os.path.join(args.output, tsv_file)
                    json_stim_metadat = os.path.join(args.output, json_stim_metadat)
                    json_stim_params = os.path.join(args.output, json_stim_params)
                    np.savetxt(tsv_file, mps, delimiter='\t')
                    with open(json_stim_metadat, 'w+') as met:
                        json.dump(metadat, met)
                    with open(json_stim_params, 'w+') as par:
                        json.dump(params, par)
                    if ("return_figures" in config and config["return_figures"]):
                        fig1.savefig(fig1_fl)
                        fig2.savefig(fig2_fl)


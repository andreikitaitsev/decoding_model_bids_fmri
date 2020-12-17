# decoding_model_bids_fmri
Repository contains code to run decoding model on FRMI data stored in the BIDS format.

The general folder structure in accordance with BIDS stanrads is assumed to be:

your_project_folder/
                   /code/audio2bidsstim
                        /data_distribution
                        /decoding
                   /raw_data/stimuli
                            /processed_stimuli
                   /processed_data
                   /lagged_data
                   /decoding_model_data
Folders in this repository in accordance with the folder hierarchy above shall be in code folder.
As follows from folder names, 
- audio2bidsstim contains code to extract features from wav files and save them in accordance with BIDS standards.
  Feature extractor may extract either Modulation Power Spectrum or Mel Spectrogram. However, all the analysis is adapted only for MPS.
- data_distribution folder contains python and bash scripts to correctly copy stimuli from raw_data/processed_stimuli to processed_data folder.
- decoding folder contains functions for data preprocessing and decoding. The high-level decoding model is contained in decoding.py file.

More detailed documentation will follow.

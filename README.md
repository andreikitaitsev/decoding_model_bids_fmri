# Decoding model BIDS fMRI
**The project is dedicated to developing pipeline to test arbitrary decoding models compatible with any [BIDS](https://bids.neuroimaging.io/) compliabt fMRI dataset.**

The code was created by Andrei Kitaitsev as part of a practical project at the Applied Neurocognitive Psychology Lab/ University of Oldenburg in 2020-2021 under the 
supervision of Moritz Boos and Dr. Arkan Al-Zubaidi. The project uses [this](https://www.nature.com/articles/sdata20143) open dataset. 

See *decoding_model_bids_fmri_example.ipynb* for step by step guidance through project scripts and functions.

See short description of the project below.

Conventional decoding models based on fMRI data exploit signal obtained by regressing BOLD response on presented stimuli convolved with hemodynamic response function (HRF).
The purpose of this project was to test a new approach to decoding based on regressing Modulation Power Spectrum (MPS) stimulus representation on “raw” BOLD fMRI in 
continuous audio stimuli paradigm (Forrest Gump dataset). Two different decoding model types were tested: one exploited purely spatial information of fMRI data (SM),
another used both spatial and temporal (STM) information by regressing MPS on lagged BOLD response. Reconstructed MPS had low correlations with the original ones, however,
original and reconstructed spectrograms obtained from the respective MPS were highly correlated, presumably due to abundance of features irrelevant for stimulus reconstruction
in the MPS. There was no univocal benefit from including temporal information into the model. Overall, the stimulus reconstruction was successful and pipeline for testing
arbitrary decoding models was created.

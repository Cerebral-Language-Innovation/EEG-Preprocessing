import mne
import numpy as np
import pandas as pd
from mne import event, pick_types
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import (ICA,create_ecg_epochs,create_eog_epochs,compute_proj_ecg,compute_proj_eog)


import pandas as pdfrom


# Read the raw data from their respective CSV file as a NumPy array*
from mne.channels import make_standard_montage

# loading 30-sec epoch single channel EEG data
mydataset = pd.read_csv("EEG_mark.csv")


#print(mydataset)

times = mydataset.iloc[0:, 0]
EEG_ch1 = mydataset.iloc[0:, 1]
EEG_ch2 = mydataset.iloc[0:, 2]
EEG_ch3 = mydataset.iloc[0:, 3]
EEG_ch4 = mydataset.iloc[0:, 4]
EEG_ch5 = mydataset.iloc[0:, 5]
EEG_ch6 = mydataset.iloc[0:, 6]
EEG_ch7 = mydataset.iloc[0:, 7]
EEG_ch8 = mydataset.iloc[0:, 8]
EEG_ch9 = mydataset.iloc[0:, 9]
EEG_ch10 = mydataset.iloc[0:, 10]
EEG_ch11 = mydataset.iloc[0:, 11]
EEG_ch12 = mydataset.iloc[0:, 12]
EEG_ch13 = mydataset.iloc[0:, 13]
EEG_ch14 = mydataset.iloc[0:, 14]
EEG_ch15 = mydataset.iloc[0:, 15]
EEG_ch16 = mydataset.iloc[0:, 16]


data = np.array([EEG_ch1, EEG_ch2, EEG_ch3, EEG_ch4, EEG_ch5,EEG_ch6, EEG_ch7, EEG_ch8, EEG_ch9, EEG_ch10, EEG_ch11, EEG_ch12, EEG_ch13, EEG_ch14, EEG_ch15, EEG_ch16])

ch_names = ["AF3",  "AF4", "F3", "Fz", "F4", "FCz", "C3", "Cz", "C4", "P3", "P4", "PO7",  "POz",  "PO8",  "O1", "O2",]
ch_types=['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg']

# Initialize an info structure
info = mne.create_info(ch_types= ch_types, sfreq = 28, ch_names=ch_names)
print(info)

raw = mne.io.RawArray(data, info)
print(raw)

montage = mne.channels.make_standard_montage('standard_1020')

mont = mne.channels.make_standard_montage("standard_1020")
selection = ["AF3",  "AF4", "F3", "Fz", "F4", "FCz", "C3", "Cz", "C4", "P3", "P4", "PO7",  "POz",  "PO8",  "O1", "O2",]


ind = [i for (i, channel) in enumerate(mont.ch_names) if channel in selection]
mont_new = mont.copy()
# Keep only the desired channels
mont_new.ch_names = [mont.ch_names[x] for x in ind]
kept_channel_info = [mont.dig[x+3] for x in ind]
# Keep the first three rows as they are the fiducial points information
mont_new.dig = mont.dig[0:3]+kept_channel_info
#mont.plot()
mont_new.plot()

raw.set_montage(mont_new)
raw.crop(tmax=60.)
picks = pick_types(raw.info, eeg=True)
ica = ICA(n_components=15, random_state=97)
ica.fit(raw)
ica.plot_sources(raw, show_scrollbars=False)
import csv
from mne.preprocessing import (ICA)
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import numpy as np

allICAs = []
subject = []
chosen = 1 #Subject to be preprocessed


def addICA(raw):
    ica = ICA(n_components=15, random_state=97)
    ica.fit(raw)
    global allICAs
    icaArray = ica.apply()
    global subject
    global choosen
    # if (subject == chosen):
    # icaArray.append(1)
    # else:
    # icaArray.append(0)
    allICAs.append(icaArray)
    print("\n ---- \n")
    print(allICAs)


def makeArray():
    global allICAs
    runs = [5, 6, 9, 10, 13, 14]
    allICAs = []
    # allICAs = [[]]
    global chosen
    for subject in range(1, 110):
        allICAs[subject].append(1) if subject == chosen else allICAs[subject].append(0)

    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    for run in runs:
        raw_fnames = eegbci.load_data(subject, run)
        raw = concatenate_raws(
            [read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(raw)
        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage)
        raw.rename_channels(lambda x: x.strip('.'))
        raw.crop(tmax=60.)
        raw.filter(14., 30.)
        picks = pick_types(raw.info, eeg=True)
        # events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        # 7
        # epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks, preload=True)
        # labels = epochs.events[:, -1] - 2;
        # filename = "eeg" + str(i) + "-"+ str(j) +".csv"
        addICA(raw)


with open('allICAs', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(allICAs)


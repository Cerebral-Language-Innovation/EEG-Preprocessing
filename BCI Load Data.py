from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci


inner_epoch = []
inner_label = []
inner_raw = []
inner_picks = []

epoch_set = []
label_set = []
raw_set = []
picks_set = []

runs_range = [5,6,9,10,13,14]

for i in range(1,110):
    for j in runs_range:
        # SELECTING THE EXPERIMENTAL DATA FROM THE EEGBCI MOTOR IMAGERY DATASET
        tmin, tmax = -1., 4.  # epochs start 1 second after experimental cue
        event_id = dict(hands=2, feet=3)  # creating a dictionary for the responses
        subject = i  # selecting the subject from the dataset
        runs = [j] # these trials correspond to when both hands or feet were imagined to be closed

        # CREATE RAW DATA STRUCTURE WITH EEGBCI DATA
        raw_names = eegbci.load_data(subject, runs)  # reading the eeg data into names
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_names])  # concatenation to create continuous data
        eegbci.standardize(raw)  # set channel names and positions to standard
        montage = make_standard_montage('standard_1005')  # use the international 10-05 system for the locations of electrodes
        raw.set_montage(montage)  # set the locations in the raw instance

        # PREPARE DATA TO BE EPOCHED
        raw.rename_channels(lambda x: x.strip('.'))  # strip channel names of "." characters
        raw.filter(7., 30.)  # apply band-pass filter
        events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))# use the annotations in the dataset to label events
        picks = pick_types(raw.info, eeg=True)  # include eeg channels
        # EPOCHING DATA
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks, preload=True) # epochs extracted from raw data
        labels = epochs.events[:, -1] - 2 # load the labels into a separate ndarraay

        inner_epoch.append(epochs)
        inner_label.append(labels)
        inner_raw.append(raw)
        inner_picks.append(picks)

    epoch_set.append(inner_epoch)
    label_set.append(inner_label)
    raw_set.append(inner_raw)
    picks_set.append(inner_picks)






## Ok, this takes an acknowledge file and converts it into
## an HDF5 dataset that the physio explorer can read.

import sys
if len(sys.argv)>1:
    fname = sys.argv[1]
else:
    print("You need to indicate a file to convert.")
    sys.exit(-1)

src = fname
#src = 'data/Sensor02.csv'

hname = "{}.hdf5".format(src.replace('.csv',''))

import json
import h5py
import pandas as pd
import numpy as np


print()
print("Converting BRAMS Bio Box data to HDF5")
print()

## gb['renames'] comes from the file configuration
bio = pd.read_csv(src,sep=',')

TIME_DIVIDER = 1000

tcol = 'Time(ms)'
bio[tcol]=bio[tcol]/TIME_DIVIDER  # express in s

tdur = max(bio[tcol])-min(bio[tcol])
SR = bio.shape[0]/tdur
print("Effective sampling rate {:.1f}".format(SR))


participant = 'a'

col_info = {
    "Time(ms)": "t",
    "PPG": "ppg",
    "ECG": "ecg",
    "Gauge": "gauge",
    "Thermistor": "therm",
    "Xaccel": "xaccel",
    "Yaccel": "yaccel",
    "Zaccel": "zaccel"
}


INVERT_ECG = True # whether to invert the ECG signal


## Drop the first few samples?
DROP_SAMPLES = 0
if DROP_SAMPLES:
    bio = bio[ DROP_SAMPLES: ]



## Create the HDF5 version
hf = h5py.File(hname, "w")
dat = hf.create_group(participant)


for col in col_info:

    print(col)
    nm = col_info[col]
    modality = nm
    p = participant
    sz = bio[col].shape[0]

    rawdata = bio[col][:]
    if INVERT_ECG and modality=='ecg':
        rawdata = -rawdata
   
    dset = hf[p].create_dataset(modality,(sz,),dtype='f',data=rawdata)
    dset.attrs['SR']=SR
    dset.attrs['participant']=p
    dset.attrs['modality']=modality

    
hf.close()
print();print("Written to {}".format(hname))



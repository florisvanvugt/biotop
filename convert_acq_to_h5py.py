

## Ok, this takes an acknowledge file and converts it into
## an HDF5 dataset that the physio explorer can read.

## ATTENTION, when using this script, you need to make sure
## that the interpretation of the channels is correct.
## That is, check the channel_contents variable below, which defines,
## in order, what each channel represents.



import sys
if len(sys.argv)>1:
    fname = sys.argv[1]
else:
    print("You need to indicate a file to convert.")
    sys.exit(-1)


src = fname


import bioread
import json
import h5py


# This file is included in bioread
data = bioread.read_file(src)

print("Channels: ")
for ch in data.channels:
    print(" {}".format(ch.name))
print()


# Channel renaming
channel_contents = [
    {'person':'a','modality':'resp'},
    {'person':'a','modality':'ppg'},
    {'person':'b','modality':'resp'},
    {'person':'b','modality':'ppg'},
    {'person':'a','modality':'ecg'},
    {'person':'a','modality':'eda'},
    {'person':'b','modality':'ecg'},
    {'person':'b','modality':'eda'},
]
participants = list(set([ ch['person'] for ch in channel_contents ]))

    
print()
print("Channels according to our labels:")
for ch in channel_contents:
    print(" {} {}".format(*ch.values()))
print()
print()



## Create the HDF5 version
hname = "{}.hdf5".format(src.replace('.acq',''))
hf = h5py.File(hname, "w")
hf.attrs['participants']=participants # set participants attribute
for p in ['a','b']:
    dat = hf.create_group(p)


SUBSAMPLING_FACTOR = 4
    
assert len(data.channels)==len(channel_contents)
for ch,info in zip(data.channels,channel_contents):
    modality = info['modality']
    p = info['person']
    rawdata = ch.data[:]
    # If we do subsampling...
    if SUBSAMPLING_FACTOR:
        rawdata = rawdata[::SUBSAMPLING_FACTOR]
    sz = rawdata.shape[0]
        
    dset = hf[p].create_dataset(modality,(sz,),dtype='f',data=rawdata)
    dset.attrs['SR']=ch.samples_per_second/SUBSAMPLING_FACTOR
    dset.attrs['participant']=p
    dset.attrs['modality']=modality
    dset.attrs['units']=ch.units

    
hf.close()
print("Written to {}".format(hname))

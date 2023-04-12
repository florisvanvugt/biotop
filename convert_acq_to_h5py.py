

## Ok, this takes an acknowledge file and converts it into
## an HDF5 dataset that the physio explorer can read.

## ATTENTION, when using this script, you need to make sure
## that the interpretation of the channels is correct.
## That is, check the channel_contents variable below, which defines,
## in order, what each channel represents.


from tkinter import filedialog as fd

import sys
if len(sys.argv)>1:
    fname = sys.argv[1]
else:

    filetypes = (
        ('Acqknowledge dataset', '*.acq'),
        ('All files', '*.*')
    )

    fname = fd.askopenfilename(
        title='Select your recording',
        initialdir='.',
        filetypes=filetypes)


if not fname:
    print("You need to indicate a file to convert.")
    sys.exit(-1)



    

src = fname


import bioread
import json
import h5py
import os


print("\n\nSource file: {}".format(os.path.basename(src)))

# This file is included in bioread
data = bioread.read_file(src)

print("Channels: ")
for ch in data.channels:
    print(" {}".format(ch.name))
print()



# EDIT THIS FOR EACH CONVERSION
participant2id = {"a":"P101","b":"P102"}
# This indicates which participant will be considered a and b in the list below.
# Do not change "a" and "b" below, they are just placeholders for the two participants


# Channel renaming
# CHECK that this is correct
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
participants = list(set([ participant2id[ch['person']] for ch in channel_contents ]))

    
print()
print("Channels according to our labels:")
for ch in channel_contents:
    print(" {} {}".format(participant2id[ch['person']],ch['modality']))
print()
print()



## Create the HDF5 version
hname = "{}.hdf5".format(src.replace('.acq',''))
hf = h5py.File(hname, "w")
hf.attrs['participants']=participants # set participants attribute
for p in ['a','b']:
    dat = hf.create_group(participant2id[p])


SUBSAMPLING_FACTOR = 1
    
assert len(data.channels)==len(channel_contents)
for ch,info in zip(data.channels,channel_contents):
    modality = info['modality']
    p = participant2id[info['person']]
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





import read_h5py

b = read_h5py.read(hname)
print(b.summary())

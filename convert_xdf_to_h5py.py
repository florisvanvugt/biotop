
import pyxdf # pip3 install pyxdf





## Ok, this takes an XDF file from LSL LabRecorder and converts it into
## an HDF5 dataset that the physio explorer can read.

## ATTENTION, this is a very case-specific script. Should not be used
## in general to convert XDF files to HDF5.


from tkinter import filedialog as fd

import sys
if len(sys.argv)>1:
    fname = sys.argv[1]
else:

    filetypes = (
        ('XDF LSL dataset', '*.xdf'),
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


import json
import h5py
import os


print("\n\nSource file: {}".format(os.path.basename(src)))

# This file is included in bioread
streams, header = pyxdf.load_xdf(fname)

print("Channels: ")
for s in streams:
    i = s['info']
    print(" {} {}".format(i['name'],i['type']))
print()

dt = header['info']['datetime']


participants = ['a']


## Create the HDF5 version
hname = "{}.hdf5".format(src.replace('.xdf',''))
hf = h5py.File(hname, "w")
hf.attrs['participants']=participants # set participants attribute
hf.attrs['date']=dt
for p in participants:
    dat = hf.create_group(p)


for s in streams:
    info = s['info']
    tp = info['type'][0].lower()
    if tp!='ecg': continue
    
    modality = tp
    p = participants[0]

    rawdata = s["time_series"].T[0] # just take the first stream
    sz = rawdata.shape[0]

    SR = info['effective_srate']
    units = info['desc'][0]['channels'][0]['channel'][0]['unit'][0]
    
    dset = hf[p].create_dataset(modality,(sz,),dtype='f',data=rawdata)
    dset.attrs['SR']=SR
    dset.attrs['participant']=p
    dset.attrs['modality']=modality
    dset.attrs['units']=units

    
hf.close()
print("Written to {}".format(hname))





import read_h5py

b = read_h5py.read(hname)
print(b.summary())




# Let's convert that to hdf5 :)


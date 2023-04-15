
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
    print(" {} {}".format(i['name'][0],i['type'][0]))
print()

dt = header['info']['datetime'][0]


participants = [ s['info']['name'][0] for s in streams ]
participants.sort()


## One issue is that in XDF, streams can start at different moments in time.
## Here we need to align them all to a common offset.
start_ts = [ min(s['time_stamps']) for s in streams ]
ONSET_T = max(start_ts)

n_samp = [ sum(s['time_stamps']>=ONSET_T) for s in streams ]
print(n_samp)
N_SAMP = min(n_samp) # take the smallest common time portion

print("---------- TIMING ----------")
print("Joint start t={}".format(ONSET_T))
print("Stream onset deltas (should not be too great)")
print([ t-ONSET_T for t in start_ts ])
print("Largest common duration: {} samples".format(N_SAMP))
print("Duration mismatches (proportion of common duration)")
print(" ".join([ "{:.4f}".format(n/N_SAMP) for n in n_samp ]))
print()

print("---------- SAMPLING RATES ----------")
nominal_sr = [ float(s['info']['nominal_srate'][0]) for s in streams ]
eff_sr     = [ float(s['info']['effective_srate']) for s in streams ]
print("Nominal sampling rates:")
print(" ".join([ "{:.2f}".format(n) for n in nominal_sr ]))
print("Effective sampling rates:")
print(" ".join([ "{:.2f}".format(n) for n in eff_sr ]))
print()


## TODO: I want to do quality assurance checks at some point too.
## Ensuring that there is not too large a gap between samples,
## that there is no 



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
    p = info['name'][0]

    t_sel = s['time_stamps']>ONSET_T
    rawdata = s["time_series"].T[0][t_sel] # just take the first stream, and only after the common starting point
    rawdata = rawdata[:N_SAMP] # take only the common chunk
    sz = rawdata.shape[0]
    assert sz==N_SAMP

    SR = float(info['nominal_srate'][0]) #info['effective_srate']
    units = info['desc'][0]['channels'][0]['channel'][0]['unit'][0]
    
    dset = hf[p].create_dataset(modality,(sz,),dtype='f',data=rawdata)
    dset.attrs['SR']=SR
    dset.attrs['participant']=p
    dset.attrs['modality']=modality
    dset.attrs['units']=units

    
hf.close()
print("-------------- Written ------------\n {}".format(hname))
print()
print()




import read_h5py

b = read_h5py.read(hname)
print(b.summary())




# Let's convert that to hdf5 :)


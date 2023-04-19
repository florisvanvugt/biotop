

# Input data from our own flavour of HDF5-dataset

import h5py
import numpy as np

class BioData:

    def __init__(self,fname):

        # This file is included in bioread
        self.hf = h5py.File(fname,'r')
        self.fname = fname

        bio = {}

        self.SR = None
        self.date = ''
        if 'participants' not in self.hf.attrs:
            print("# Error, missing participants attribute in the root.")
        if 'date' in self.hf.attrs:
            self.date = self.hf.attrs['date']
        self.participants = self.hf.attrs['participants']
        self.channels = []
        self.channels_by_type = {}
        for p in self.participants:
            for ch in self.hf[p].keys():
                nm = "{}/{}".format(p,ch)
                dset = self.hf[p][ch]
                if self.SR:
                    if dset.attrs['SR']!=self.SR:
                        print("## ERROR, sampling rate {} differs from that of other channels. Not currently implemented.")
                else:
                    self.SR=dset.attrs['SR']
                assert self.SR==dset.attrs['SR']
                bio[nm]=dset ##np.array(dset[:]) # convert into numpy array just to be sure
                mod = dset.attrs['modality']
                self.channels.append(nm)
                self.channels_by_type[mod] = self.channels_by_type.get(mod,[])+[nm]

        bio['t']=np.arange(dset.shape[0])/self.SR

        self.bio = bio
        self.preprocessed = {}



    def get_channels(self):
        return self.channels

    def get_ecg_channels(self):
        return self.channels_by_type['ecg']



    def get_participants(self):
        part = self.participants
        part.sort()
        return part



    def summary(self):
        ret = "Summary of {}\n".format(self.fname)
        if self.date:
            ret += '[ date : {} ]\n'.format(self.date)
        for p in self.participants:
            ret += "\nParticipant '{}'\n".format(p)
            part = self.hf[p]
            for chn in part:
                ch = part[chn]
                nsamp = ch.size
                frq = ch.attrs['SR']
                dur = nsamp/frq
                ret += "∟ channel {} [ modality {} in {} ] {} samples @ {:.1f} Hz = {:.1f} s\n".format(
                    chn,
                    ch.attrs['modality'],
                    ch.attrs.get('units','a.u.'),
                    nsamp,
                    frq,
                    dur
                )
        return (ret)





def read(fname):
    bio = BioData(fname)
    return bio




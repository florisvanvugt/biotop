

## Following roughly practices from
## https://www.kaggle.com/code/nelsonsharma/ecg-02-ecg-signal-pre-processing

import numpy as np

from scipy.io import loadmat
from scipy import signal
from scipy.signal import medfilt
import pywt
from pywt import wavedec

from ecgdetectors import Detectors

import neurokit2



def denoise_signal(X, dwt_transform, dlevels, cutoff_low, cutoff_high):
    coeffs = wavedec(X, dwt_transform, level=dlevels)   # wavelet transform 'bior4.4'
    # scale 0 to cutoff_low 
    for ca in range(0,cutoff_low):
        coeffs[ca]=np.multiply(coeffs[ca],[0.0])
    # scale cutoff_high to end
    for ca in range(cutoff_high, len(coeffs)):
        coeffs[ca]=np.multiply(coeffs[ca],[0.0])
    Y = pywt.waverec(coeffs, dwt_transform) # inverse wavelet transform
    return Y  



def get_median_filter_width(sampling_rate, duration):
    res = int( sampling_rate*duration )
    res += ((res%2) - 1) # needs to be an odd number
    return res



def filter_signal(X,mfa):
    X0 = X  #read orignal signal
    for mi in range(0,len(mfa)):
        X0 = medfilt(X0,mfa[mi]) # apply median filter one by one on top of each other
    X0 = np.subtract(X,X0)  # finally subtract from orignal signal
    return X0



FILTER = True  # possibly better without?


def preprocess(biodata,gb,fields=None):
    print("\nECG preprocessing using custom & py-ecg-detectors strategy...")
    res = {}
    bio = biodata.bio
    SR = biodata.SR

    todo = list(gb['ecg_preprocess'].keys())
    if fields: todo = fields
    
    for ecg_chan in todo:

        ecg_target = gb['ecg_preprocess'][ecg_chan]

        signal = gb['bio'][ecg_chan]
        ## Denoising
        ##print("Denoising...")
        if FILTER:
            #signal_den = denoise_signal(signal,'bior4.4', 9 , 1 , 7) #<--- trade off - the less the cutoff - the more R-peak morphology is lost
            # baseline fitting by filtering
            # === Define Filtering Params for Baseline fitting Leads======================
            #ms_flt_array = [0.2,0.6]    #<-- length of baseline fitting filters (in seconds)
            #mfa = np.zeros(len(ms_flt_array), dtype='int')
            #for i in range(0, len(ms_flt_array)):
            #     mfa[i] = get_median_filter_width(SR,ms_flt_array[i])
            # signal_flt = filter_signal(signal_den,mfa)

            # Check shape
            #n_orig = signal.shape[0]
            #n_filt = signal_flt.shape[0]
            #print(n_orig,n_filt)
            METHOD = 'engzeemod2012'
            #METHOD = 'neurokit2'
            #METHOD
            signal_flt = neurokit2.ecg_clean(signal,sampling_rate=SR,method=METHOD)
            
        else:
            signal_flt = signal
            

        print("--> Detecting {}...".format(ecg_chan))
        detectors = Detectors(int(round(SR)))
        try:
            r_peaks = detectors.engzee_detector(signal_flt)  # note that the py-ecg-detectors makers appear to suggest we should use unfiltered ECG
        except:
            r_peaks = []

        biodata.bio[ecg_target] = signal_flt
        biodata.preprocessed[ecg_target] = {
            'ecg_peaks' : r_peaks,
            'ecg_t': np.arange(0, len(signal_flt)/SR, 1/SR)}
        
    return res

    

ALPHA = .5

def draw(ax,biodata,drawrange,ecg_target,gb):

    tmin,tmax = drawrange
    
    # Plot on axis
    SR = biodata.SR

    #check if there's a rounding error causing differing lengths of plotx and signal
    ecg = biodata.bio[ecg_target] # gb['ecg_clean']

    prep = biodata.preprocessed[ecg_target]
    plot_t = prep['ecg_t']
    tsels = (plot_t>=tmin) & (plot_t<=tmax)

    peaklist = [ (p,p/SR,ecg[p]) for p in prep['ecg_peaks'] ]
    
    #ax.plot(plotx, working_data['hr'], label='heart rate signal', zorder=-10)
    for _,t,_ in peaklist: # this helps to avoid overplotting
        if t>=tmin and t<=tmax:
            ax.axvline(t,color='gray',zorder=-99,lw=1)
    ax.plot(plot_t[tsels],
            ecg[tsels],
            label='cleaned',
            zorder=-10)
    for _,t,y in peaklist: # this helps to avoid overplotting
        if t>=tmin and t<=tmax:
            ax.plot(t,y,'o',mfc='green',mec='green',alpha=ALPHA) # accepted peaks, label='BPM') #:%.2f' %(measures['bpm']))

    ax.set_ylabel('{}\n[ecg-detect]'.format(ecg_target))




def get_ylim(tstart,tend,gb):

    sigs = []

    ecg_preps = gb['ecg_preprocess'].values()
    biodata = gb['biodata']
    for ecg_target in ecg_preps:

        ecg = biodata.bio[ecg_target] # gb['ecg_clean']
        prep = biodata.preprocessed[ecg_target]
        t_ecg = prep['ecg_t']
        tsel_ecg = (t_ecg>tstart) & (t_ecg<tend)
        sig = biodata.bio[ecg_target]
        sig = sig[ tsel_ecg ] # make the selection
        sigs.append(sig)

    sig = np.concatenate(sigs)
        
    if len(sig):
        mn,mx = np.min(sig),np.max(sig)
        # add some padding on the sides
        pad = .05*(mx-mn)
        return (mn-pad,mx+pad)
    else:
        return (-1,1)


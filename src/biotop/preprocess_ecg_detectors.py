

## Following roughly practices from
## https://www.kaggle.com/code/nelsonsharma/ecg-02-ecg-signal-pre-processing

import numpy as np

from scipy.io import loadmat
from scipy import signal
from scipy.signal import medfilt

from ecgdetectors import Detectors

import neurokit2



def denoise_signal(X, dwt_transform, dlevels, cutoff_low, cutoff_high):
    import pywt
    from pywt import wavedec
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
    res = {}
    bio = biodata.bio
    SR = biodata.SR

    todo = list(gb['ecg_preprocess'].keys())
    if fields: todo = fields

    for ecg_chan in todo:

        ecg_target = gb['ecg_preprocess'][ecg_chan]

        signal = gb['bio'][ecg_chan]
        signal_flt = signal
        
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
            print("Filtering ECG signal using {} procedure in neurokit2".format(METHOD))
            signal_flt = neurokit2.ecg_clean(signal,sampling_rate=SR,method=METHOD)
            
        biodata.bio[ecg_target] = signal_flt



def peak_detect(signal,SR,detector):

    print("--> Peak detection using {}".format(detector))
    detectors = Detectors(int(round(SR)))

    d = ['hamilton','christov','engzee','pan_tompkins','swt','two_average']

    det = None
    if detector=='hamilton': det=detectors.hamilton_detector
    if detector=='christov': det=detectors.christov_detector
    if detector=='engzee': det=detectors.engzee_detector
    if detector=='pan_tompkins': det=detectors.pan_tompkins_detector
    if detector=='swt': det=detectors.swt_detector
    if detector=='two_average': det=detectors.two_average_detector
    if detector=='matched_filter': det=detectors.matched_filter_detector
    if detector=='wqrs': det=detectors.wqrs_detector
    if not det:
        print("# ERROR, unknown detector.")
        return []
    
    try:
        r_peaks = det(signal)  # note that the py-ecg-detectors makers appear to suggest we should use unfiltered ECG
    except:
        r_peaks = []

    return r_peaks





